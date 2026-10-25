import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Union, Optional
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, NEW_MODEL_PATH
import models.new_model.inference as inference

logger = logging.getLogger(__name__)

class FetalSegmentationPredictor:
    """
    Production-ready Inference Engine for Fetal Ultrasound Segmentation using best_attunet.pth.
    """
    CLASS_NAMES = ["Background", "Fetal Abdomen"]
    NUM_CLASSES = 2
    
    def __init__(
        self, 
        model_path: Optional[Union[str, Path]] = None, 
        device: Optional[str] = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_path is None:
            model_path = NEW_MODEL_PATH if NEW_MODEL_PATH.exists() else MODELS_DIR / "new_model" / "best_attunet.pth"
            
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        
        self.load_weights()

    def load_weights(self) -> bool:
        """Loads trained model weights from disk checkpoint."""
        if self.model_path.exists():
            try:
                self.model, dev = inference.load_model(str(self.model_path), device=self.device)
                self.device = str(dev)
                self.is_loaded = True
                logger.info(f"[FetalPredictor] Successfully loaded AttUNet checkpoint from {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"[FetalPredictor] Failed to load checkpoint from {self.model_path}: {e}")
                self.is_loaded = False
                return False
        else:
            logger.info(f"[FetalPredictor] Note: No model checkpoint found at {self.model_path}.")
            self.is_loaded = False
            return False

    def predict_image_array(
        self, 
        image_np: np.ndarray, 
        target_size: Optional[Tuple[int, int]] = (320, 416),
        apply_softmax: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs segmentation prediction on an RGB or Grayscale NumPy image array using AttUNet.
        Returns:
            pred_mask: (H, W) NumPy array with class label (0=Background, 1=Fetal Abdomen)
            prob_map: (H, W) probability map
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"No trained fetal segmentation checkpoint is loaded from {self.model_path}.")

        import cv2
        img = inference.to_uint8(np.asarray(image_np))
        if img.ndim == 3:
            img = (cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2GRAY)
                   if img.shape[2] >= 3 else img[..., 0])
        orig_h, orig_w = img.shape

        in_h, in_w = target_size if target_size else (inference.H, inference.W)
        small = cv2.resize(img, (in_w, in_h), interpolation=cv2.INTER_AREA)
        device = torch.device(self.device)
        pr = inference.predict_probs(self.model, device, small[None])[0]
        
        m_small = inference.postprocess(pr > 0.5)
        if not m_small.any():
            pred_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        else:
            pred_mask = (cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
            
        prob_map = cv2.resize(pr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return pred_mask, prob_map

    def predict_npy_file(self, npy_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Loads a dataset .npy file, runs segmentation inference, and returns structured result dictionary.
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(f"File not found: {npy_path}")
            
        data_dict = np.load(npy_path, allow_pickle=True).item()
        
        image_np = data_dict["image"]
        structures = data_dict.get("structures", {})
        
        pred_mask, prob_map = self.predict_image_array(image_np)
        
        return {
            "filename": npy_path.name,
            "image": image_np,
            "ground_truth_structures": structures,
            "predicted_mask": pred_mask,
            "probabilities": prob_map,
            "unique_classes": np.unique(pred_mask).tolist()
        }

if __name__ == "__main__":
    predictor = FetalSegmentationPredictor()
    dummy_img = np.random.randint(0, 255, (768, 1024, 3), dtype=np.uint8)
    mask, probs = predictor.predict_image_array(dummy_img)
    print("Inference Successful! Mask Shape:", mask.shape, "Classes Predicted:", np.unique(mask))
