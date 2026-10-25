import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, CTGAN_EPOCHS, CTGAN_BATCH_SIZE, RANDOM_STATE

logger = logging.getLogger(__name__)

class CTGANDigitalTwin:
    """
    CTGAN-based Generative Digital Twin for Maternal Clinical Data Augmentation.
    Follows methodology of 'Data Augmentation Via Digital Twins...'.
    Generates synthetic maternal cohorts for small data contexts.
    """
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or (MODELS_DIR / "ctgan_maternal_twin.joblib")
        self.ctgan_model = None
        self.is_fitted = False
        self.load()

    def save(self):
        """Saves trained CTGAN model checkpoint."""
        if self.ctgan_model is not None and self.is_fitted:
            import joblib
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.ctgan_model, self.model_path)
            logger.info(f"Saved CTGAN model to {self.model_path}")

    def load(self) -> bool:
        """Loads CTGAN model checkpoint if exists."""
        if self.model_path.exists():
            try:
                import joblib
                import torch

                # Handle loading CUDA-saved PyTorch tensors on CPU devices
                if not torch.cuda.is_available():
                    orig_torch_load = torch.load
                    def cpu_torch_load(*args, **kwargs):
                        if 'map_location' not in kwargs:
                            kwargs['map_location'] = torch.device('cpu')
                        return orig_torch_load(*args, **kwargs)

                    try:
                        torch.load = cpu_torch_load
                        self.ctgan_model = joblib.load(self.model_path)
                    finally:
                        torch.load = orig_torch_load
                else:
                    self.ctgan_model = joblib.load(self.model_path)

                self.is_fitted = True
                logger.info(f"Loaded CTGAN model from {self.model_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load CTGAN model: {e}")
                return False
        return False

    def train(self, data_path: Optional[Path] = None, epochs: int = CTGAN_EPOCHS, batch_size: int = CTGAN_BATCH_SIZE, force: bool = False, exclude_columns: Optional[List[str]] = None) -> bool:
        """
        Trains CTGAN on processed maternal clinical data.
        """
        if self.is_fitted and not force:
            logger.info(f"CTGAN model is already loaded from {self.model_path}. Skipping training (pass force=True or set RETRAIN_CTGAN=1 to retrain).")
            return True

        try:
            from ctgan import CTGAN
        except ImportError:
            logger.warning("ctgan package not installed. Installing or using bootstrap fallback.")
            return False

        data_path = data_path or (PROCESSED_DIR / "train_clean.csv")
        if not data_path.exists():
            logger.error(f"Training data not found at {data_path}")
            return False
            
        logger.info(f"Loading training data from {data_path} for CTGAN...")
        df_train = pd.read_csv(data_path)
        
        # Ensure CTGAN trains strictly on the clean, verified early booking predictors + target
        from config import EARLY_BOOKING_PREDICTORS, TARGET_COLUMN
        features_to_model = [col for col in EARLY_BOOKING_PREDICTORS if col in df_train.columns]
        if TARGET_COLUMN in df_train.columns and TARGET_COLUMN not in (exclude_columns or []):
            features_to_model.append(TARGET_COLUMN)
            
        df_train = df_train[features_to_model].copy()
        excluded_columns = [column for column in (exclude_columns or []) if column in df_train.columns]
        if excluded_columns:
            df_train = df_train.drop(columns=excluded_columns)
        
        # CTGAN does not allow missing values in training data - fill nulls
        for col in df_train.columns:
            if df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category':
                df_train[col] = df_train[col].fillna("Missing")
            else:
                med_val = df_train[col].median()
                df_train[col] = df_train[col].fillna(med_val if not pd.isna(med_val) else 0.0)
        
        # Identify discrete categorical columns
        discrete_columns = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Ensure batch_size is multiple of CTGAN pac (10) and dataset size is divisible by pac (10)
        pac = 10
        batch_size = (batch_size // pac) * pac
        if batch_size < pac:
            batch_size = pac
            
        valid_len = (len(df_train) // pac) * pac
        df_train = df_train.iloc[:valid_len]
        
        logger.info(f"Training CTGAN model for {epochs} epochs on {len(df_train)} rows (batch_size={batch_size}, pac={pac})...")
        self.ctgan_model = CTGAN(epochs=epochs, batch_size=batch_size, pac=pac, verbose=False)
        self.ctgan_model.fit(df_train, discrete_columns)


        
        self.is_fitted = True
        self.save()
        logger.info("CTGAN Digital Twin model training complete.")
        return True

    def generate_synthetic_cohort(self, num_samples: int = 500, save_path: Optional[Path] = None, include_fallback_label: bool = True) -> pd.DataFrame:
        """
        Generates synthetic maternal patient profiles for data augmentation.
        """
        save_path = save_path or (SYNTHETIC_DIR / "synthetic_maternal_patients.csv")
        
        if self.ctgan_model is not None and self.is_fitted:
            logger.info(f"Generating {num_samples} synthetic maternal profiles via CTGAN...")
            df_synth = self.ctgan_model.sample(num_samples)
            # Post-process to ensure biologically plausible bounds per NICE guidelines
            if "AgeAtStartOfSpell" in df_synth.columns:
                df_synth["AgeAtStartOfSpell"] = df_synth["AgeAtStartOfSpell"].clip(15.0, 55.0).round(1)
            if "Body Mass Index at Booking" in df_synth.columns:
                df_synth["Body Mass Index at Booking"] = df_synth["Body Mass Index at Booking"].clip(15.0, 60.0).round(2)
            if "SystolicBloodPressureCuff" in df_synth.columns:
                df_synth["SystolicBloodPressureCuff"] = df_synth["SystolicBloodPressureCuff"].clip(70.0, 200.0).round(1)
            if "Diastolic Blood Pressure" in df_synth.columns:
                df_synth["Diastolic Blood Pressure"] = df_synth["Diastolic Blood Pressure"].clip(40.0, 130.0).round(1)
            if "Gravida" in df_synth.columns:
                df_synth["Gravida"] = df_synth["Gravida"].clip(1, 15).round(0).astype(int)
            if "Parity" in df_synth.columns:
                df_synth["Parity"] = df_synth["Parity"].clip(0, 12).round(0).astype(int)
            if "No_Of_previous_Csections" in df_synth.columns:
                df_synth["No_Of_previous_Csections"] = df_synth["No_Of_previous_Csections"].clip(0, 8).round(0).astype(int)
            if "Gestation at booking (Weeks)" in df_synth.columns:
                df_synth["Gestation at booking (Weeks)"] = df_synth["Gestation at booking (Weeks)"].clip(6.0, 36.0).round(1)
        else:
            # Fallback: Bootstrapping / Gaussian noise sampling from train_clean if CTGAN weights not pre-fitted
            logger.info(f"Generating {num_samples} synthetic maternal profiles via clean feature distribution sampling...")
            train_clean_path = PROCESSED_DIR / "train_clean.csv"
            if train_clean_path.exists():
                df_real = pd.read_csv(train_clean_path)
                df_synth = df_real.sample(n=num_samples, replace=True, random_state=RANDOM_STATE).copy()
                
                # Add slight noise to numeric columns for variation
                num_cols = df_synth.select_dtypes(include=[np.number]).columns
                for col in num_cols:
                    std = df_real[col].std()
                    if pd.notna(std) and std > 0:
                        noise = np.random.normal(0, std * 0.05, size=len(df_synth))
                        df_synth[col] = df_synth[col] + noise
            else:
                df_synth = pd.DataFrame()

        if not include_fallback_label and not df_synth.empty:
            from config import TARGET_COLUMN
            df_synth = df_synth.drop(columns=[TARGET_COLUMN], errors="ignore")

        if not df_synth.empty:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_synth.to_csv(save_path, index=False)
            logger.info(f"Saved synthetic cohort to {save_path}")

        return df_synth

    def simulate_counterfactual(self, patient_profile: Dict[str, Any], feature_modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates counterfactual 'what-if' scenarios (e.g. what if maternal BMI or Glucose changes?).
        """
        simulated = patient_profile.copy()
        for k, v in feature_modifications.items():
            simulated[k] = v
        return simulated

if __name__ == "__main__":
    twin = CTGANDigitalTwin()
    synth_df = twin.generate_synthetic_cohort(num_samples=50)
    print("Synthetic cohort shape:", synth_df.shape)
