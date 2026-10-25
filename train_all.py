import os
import sys
import logging
from pathlib import Path

# Setup logging to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_pipeline.log", mode="w")
    ]
)
logger = logging.getLogger("TrainPipeline")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent))

# Import pipeline steps
# Import pipeline steps
from preprocessing.preprocess import run_preprocessing_pipeline
from digital_twin.ctgan_twin import CTGANDigitalTwin
from digital_twin.evaluate_synthetic import SyntheticDataEvaluator
from digital_twin.neural_ode import train_neural_ode
from models.maternal_predictor import MaternalGDMPredictor
from fetal.predict import FetalSegmentationPredictor
from config import MODELS_DIR, NEW_MODEL_PATH

def main():
    logger.info("=== STARTING MDT-GEN INTEGRATED MODEL TRAINING PIPELINE ===")
    
    # Step 1: Preprocessing & Data Cleaning
    logger.info("--- Step 1: Data Preprocessing ---")
    try:
        run_preprocessing_pipeline()
        logger.info("Preprocessing complete.")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}", exc_info=True)
        sys.exit(1)
        
    # Step 2: CTGAN Model Training
    logger.info("--- Step 2: CTGAN Generative Digital Twin Training ---")
    try:
        ctgan_twin = CTGANDigitalTwin()
        retrain_ctgan = os.environ.get("RETRAIN_CTGAN", "0").lower() in ("1", "true", "yes")
        if ctgan_twin.is_fitted and not retrain_ctgan:
            logger.info(f"Found trained CTGAN checkpoint at {ctgan_twin.model_path}. Skipping CTGAN training (set RETRAIN_CTGAN=1 to force retraining).")
            logger.info("Generating synthetic patient cohort from loaded CTGAN model...")
            ctgan_twin.generate_synthetic_cohort(num_samples=500)
        else:
            epochs = int(os.environ.get("CTGAN_EPOCHS", 50))
            logger.info(f"Starting CTGAN training for {epochs} epochs...")
            success = ctgan_twin.train(epochs=epochs, batch_size=256, force=True)
            if success:
                logger.info("CTGAN training complete.")
                logger.info("Generating synthetic patient cohort...")
                ctgan_twin.generate_synthetic_cohort(num_samples=500)
            else:
                logger.warning("CTGAN training skipped or fell back. Generating cohort via fallback distribution sampling.")
                ctgan_twin.generate_synthetic_cohort(num_samples=500)
    except Exception as e:
        logger.error(f"CTGAN training failed: {e}", exc_info=True)
        sys.exit(1)
        
    # Step 3: Synthetic Data Utility Evaluation
    logger.info("--- Step 3: Synthetic Data Utility Evaluation ---")
    try:
        evaluator = SyntheticDataEvaluator()
        eval_results = evaluator.run_evaluation()
        logger.info("Synthetic data evaluation complete.")
        logger.info(f"Preservation MAE: {eval_results.get('correlation_preservation_mae', 'N/A')}")
        logger.info(f"Privacy Risk level: {eval_results.get('privacy_leakage', {}).get('privacy_risk', 'N/A')}")
    except Exception as e:
        logger.error(f"Synthetic data evaluation failed: {e}", exc_info=True)
        
    # Step 4: Neural ODE Longitudinal Trajectory Training
    logger.info("--- Step 4: Neural ODE Trajectory Generator Training ---")
    try:
        ode_ckpt = MODELS_DIR / "neural_ode_maternal_gdm.pth"
        retrain_ode = os.environ.get("RETRAIN_ODE", "0").lower() in ("1", "true", "yes")
        if ode_ckpt.exists() and not retrain_ode:
            logger.info(f"Found trained Neural ODE checkpoint at {ode_ckpt}. Skipping Neural ODE training (set RETRAIN_ODE=1 to force retraining).")
        else:
            epochs_ode = int(os.environ.get("ODE_EPOCHS", 100))
            model_ode, info_ode = train_neural_ode(epochs=epochs_ode)
            logger.info("Neural ODE training complete.")
    except Exception as e:
        logger.error(f"Neural ODE training failed: {e}", exc_info=True)
        sys.exit(1)
        
    # Step 5: Stacking Ensemble Classifier Training
    logger.info("--- Step 5: Stacking Ensemble Training ---")
    try:
        predictor = MaternalGDMPredictor()
        eval_metrics = predictor.train_and_evaluate()
        logger.info("Validated GDM and macrosomia booking models trained (reports/validated/).")
        logger.info(f"GDM held-out AUROC: {eval_metrics['roc_auc']:.3f} | Brier: {eval_metrics['brier']:.4f}")
        logger.info(f"Macrosomia held-out AUROC: {eval_metrics['macrosomia_roc_auc']:.3f} | Brier: {eval_metrics['macrosomia_brier']:.4f}")
    except Exception as e:
        logger.error(f"Stacking Ensemble training failed: {e}", exc_info=True)
        sys.exit(1)
        
    # Step 6: Fetal Ultrasound Attention U-Net (ACOUSLIC-AI) - verify the checkpoint loads.
    # The model is trained on Kaggle (kaggle_acouslic_attention_unet.py); AC accuracy is
    # evaluated with fetal/evaluate_ac_accuracy.py on held-out ACOUSLIC-AI cases.
    logger.info("--- Step 6: Fetal Ultrasound Attention U-Net (ACOUSLIC-AI) ---")
    unet_ckpt = NEW_MODEL_PATH
    fetal_predictor = FetalSegmentationPredictor(model_path=unet_ckpt) if unet_ckpt.exists() else None
    if fetal_predictor is not None and fetal_predictor.is_loaded:
        logger.info(f"Attention U-Net loaded from {unet_ckpt}.")
    else:
        logger.warning(f"No loadable Attention U-Net checkpoint at {unet_ckpt}; export best_attunet.pth from Kaggle.")

    logger.info("=== ALL PIPELINE STAGES COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
