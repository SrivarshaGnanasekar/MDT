import sys
import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_PATH, PROCESSED_DIR, MODELS_DIR, TARGET_COLUMN, COLS_TO_DROP, 
    LAB_ERROR_COLS, STATIC_FEATURES, DYNAMIC_FEATURES, OUTCOME_FEATURES,
    TEST_SIZE, RANDOM_STATE
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MaternalPreprocessor:
    def __init__(self):
        self.fitted = False
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_encoders = {}
        
        # Columns tracked
        self.numeric_features = []
        self.categorical_features = []
        
    @staticmethod
    def clean_lab_value(val: Any) -> float:
        """Parse a clinical value that might contain text errors or clinical bounds."""
        if pd.isna(val):
            return np.nan
        
        val_str = str(val).strip()
        
        # Handle clinical bounds like '<10.0' or '<10'
        if val_str.startswith('<'):
            try:
                # Remove '<' and parse float
                return float(val_str.replace('<', '').strip())
            except ValueError:
                pass
                
        # Extract numerical pattern (int or float)
        import re
        match = re.search(r'(\d+\.?\d*)', val_str)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return np.nan
        return np.nan

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies basic data cleaning and formats column types."""
        logger.info("Cleaning raw dataset...")
        df_clean = df.copy()
        
        # 1. Drop unnecessary columns
        cols_present_to_drop = [col for col in COLS_TO_DROP if col in df_clean.columns]
        df_clean = df_clean.drop(columns=cols_present_to_drop)
        
        # 2. Clean laboratory columns
        for col in LAB_ERROR_COLS:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].apply(self.clean_lab_value)

        # Condense Height and Weight into BMI if BMI has missing values
        if "Body Mass Index at Booking" in df_clean.columns and "WeightMeasured" in df_clean.columns and "Height" in df_clean.columns:
            calc_bmi = df_clean["WeightMeasured"] / ((df_clean["Height"] / 100.0) ** 2)
            df_clean["Body Mass Index at Booking"] = df_clean["Body Mass Index at Booking"].fillna(calc_bmi)
                
        # 3. Clean and map target variable
        if TARGET_COLUMN in df_clean.columns:
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({"Yes": 1, "No": 0})
            # Drop rows where target is missing if any
            df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
            
        # 4. Fill missing values in categorical fields with logical placeholders
        cat_cols = df_clean.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in ["Risk Factors", "AntenatalMedicalFactors", "PreviousObstetricHistory", "PrimaryIndicationforCaesarean", "Category Caesarean Section"]:
                df_clean[col] = df_clean[col].fillna("None")
            elif col in ["GlucoseToleranceTest", "FolicAcidDose"]:
                df_clean[col] = df_clean[col].fillna("Unknown")
            else:
                df_clean[col] = df_clean[col].fillna("Unknown")
                
        # 5. Fill missing values in binary variables with 'No'
        binary_cols = ["Obese?", "Severely Premature?", "Presence of meconium", "Shoulder Dystocia", "SCBU admission (Yes/NO)", "Still_Birth"]
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("No")
                
        # 6. Standardize binary text
        for col in binary_cols + ["Sex"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
                # Map to binary string
                if col in binary_cols:
                    df_clean[col] = df_clean[col].map({"YES": "YES", "NO": "NO", "TRUE": "YES", "FALSE": "NO"}).fillna("NO")
                
        logger.info(f"Cleaned dataset shape: {df_clean.shape}")
        return df_clean

    def split_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Splits features into numerical and categorical lists for predictive model."""
        predictor_cols = STATIC_FEATURES + DYNAMIC_FEATURES
        # Filter existing features
        predictor_cols = [c for c in predictor_cols if c in df.columns and c != TARGET_COLUMN]
        
        num_features = []
        cat_features = []
        
        for col in predictor_cols:
            # Check data type
            if df[col].dtype in [np.float64, np.int64]:
                # If binary represented as number, check unique
                if df[col].nunique() <= 2 and df[col].dropna().isin([0, 1]).all():
                    cat_features.append(col)
                else:
                    num_features.append(col)
            else:
                cat_features.append(col)
                
        return num_features, cat_features

    def fit(self, train_df: pd.DataFrame):
        """Fits imputers, scalers, and encoders on the training set."""
        logger.info("Fitting preprocessing components...")
        
        # Split features
        self.numeric_features, self.categorical_features = self.split_features(train_df)
        
        # Fit numerical imputer
        if self.numeric_features:
            train_num = train_df[self.numeric_features]
            self.num_imputer.fit(train_num)
            
            # Fit scaler
            imputed_num = self.num_imputer.transform(train_num)
            self.scaler.fit(imputed_num)
            
        # Fit categorical encoders
        for col in self.categorical_features:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoder.fit(train_df[[col]].astype(str))
            self.cat_encoders[col] = encoder
            
        self.fitted = True
        logger.info("Preprocessing components successfully fitted.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a dataframe using the fitted preprocessors."""
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
            
        df_input = df.copy()
        for col in self.numeric_features:
            if col not in df_input.columns:
                df_input[col] = np.nan
        for col in self.categorical_features:
            if col not in df_input.columns:
                df_input[col] = "Unknown"
                
        df_out = pd.DataFrame(index=df_input.index)
        
        # 1. Process numerical columns
        if self.numeric_features:
            num_data = df_input[self.numeric_features]
            imputed_num = self.num_imputer.transform(num_data)
            scaled_num = self.scaler.transform(imputed_num)
            
            # Recreate DataFrame
            df_num = pd.DataFrame(scaled_num, columns=self.numeric_features, index=df_input.index)
            df_out = pd.concat([df_out, df_num], axis=1)
            
        # 2. Process categorical columns (One-Hot Encoding)
        for col in self.categorical_features:
            encoder = self.cat_encoders[col]
            encoded_data = encoder.transform(df_input[[col]].astype(str))
            encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            
            df_cat = pd.DataFrame(encoded_data, columns=encoded_cols, index=df_input.index)
            df_out = pd.concat([df_out, df_cat], axis=1)
            
        return df_out

    def save(self, path: Path):
        """Saves the preprocessor model to disk."""
        joblib.dump(self, path)
        logger.info(f"Saved preprocessor to {path}")

    @classmethod
    def load(cls, path: Path) -> 'MaternalPreprocessor':
        """Loads a preprocessor model from disk."""
        preprocessor = joblib.load(path)
        imputer = preprocessor.num_imputer
        if not hasattr(imputer, "_fill_dtype") and hasattr(imputer, "_fit_dtype"):
            imputer._fill_dtype = imputer._fit_dtype
        return preprocessor


def run_preprocessing_pipeline():
    logger.info("Starting preprocessing pipeline...")
    
    # 1. Read Raw Data
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="V2.0.1", header=1)
    
    preprocessor = MaternalPreprocessor()
    
    # 2. Clean the full dataset
    df_cleaned = preprocessor.clean_dataset(df)
    
    # 3. Train-Test Split (on cleaned data)
    # We do a patient-level split (UID is dropped, but index represents individual rows)
    train_df, test_df = train_test_split(df_cleaned, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_cleaned[TARGET_COLUMN])
    
    logger.info(f"Split completed: Train shape {train_df.shape}, Test shape {test_df.shape}")
    
    # Save the cleaned (but not one-hot encoded or scaled) train/test sets
    # These are needed for CTGAN digital twin training and dynamic visualization
    train_df.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_clean.csv", index=False)
    logger.info("Saved clean train/test splits for CTGAN.")
    
    # 4. Fit preprocessor on the train set (only using predictors: Static + Dynamic features)
    preprocessor.fit(train_df)
    
    # 5. Transform Train and Test sets for Machine Learning modeling
    # Features (X) and Target (y)
    X_train_encoded = preprocessor.transform(train_df)
    X_test_encoded = preprocessor.transform(test_df)
    
    y_train = train_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]
    
    # Combine into dataframes to save
    train_ml = X_train_encoded.copy()
    train_ml[TARGET_COLUMN] = y_train.values
    
    test_ml = X_test_encoded.copy()
    test_ml[TARGET_COLUMN] = y_test.values
    
    # Save processed ML files
    train_ml.to_csv(PROCESSED_DIR / "train_ml.csv", index=False)
    test_ml.to_csv(PROCESSED_DIR / "test_ml.csv", index=False)
    
    # NOTE: the fitted MaternalPreprocessor is no longer saved. The dashboard uses the
    # validated bundles in models/validated/ (models/train_validated_booking_models.py),
    # which carry their own fold-safe preprocessor.
    
    logger.info("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
