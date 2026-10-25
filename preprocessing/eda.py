import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_PATH, REPORTS_DIR, TARGET_COLUMN, STATIC_FEATURES, DYNAMIC_FEATURES, OUTCOME_FEATURES, LAB_ERROR_COLS

def run_eda():
    print("--- Starting Exploratory Data Analysis ---")
    
    # Load dataset
    print(f"Loading raw dataset from {RAW_DATA_PATH}...")
    df = pd.read_excel(RAW_DATA_PATH, sheet_name="V2.0.1", header=1)
    print(f"Dataset shape: {df.shape}")
    
    # Create reports directory if not exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Missing Value Analysis
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing_counts, "Percentage (%)": missing_pct})
    missing_df = missing_df.sort_values(by="Percentage (%)", ascending=False)
    
    print("\n--- Missing Value Summary (Top 15) ---")
    print(missing_df.head(15).to_string())
    
    # Save missing value plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_df.index[:20], y=missing_df["Percentage (%)"][:20], palette="viridis")
    plt.xticks(rotation=90)
    plt.title("Percentage of Missing Values per Feature (Top 20)")
    plt.ylabel("Missing Percentage (%)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "missing_values.png", dpi=150)
    plt.close()
    
    # 2. Class Imbalance (Target Column)
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False)
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100
    
    print(f"\n--- Target Label '{TARGET_COLUMN}' Distribution ---")
    for val, count, pct in zip(target_counts.index, target_counts.values, target_pct.values):
        print(f"  {val}: {count} ({pct:.2f}%)")
        
    plt.figure(figsize=(6, 5))
    sns.countplot(x=TARGET_COLUMN, data=df, palette="coolwarm")
    plt.title("Gestational Diabetes Class Distribution")
    plt.ylabel("Patient Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_imbalance.png", dpi=150)
    plt.close()
    
    # 3. Clean Lab Error Cols temporarily for Correlation analysis
    df_temp = df.copy()
    for col in LAB_ERROR_COLS:
        if col in df_temp.columns:
            # Strip any text and convert to numeric
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
            
    # Also parse 'VitaminDlevelblood' '<10.0' or other things
    # We will do a full clean in preprocess.py, this is just for corr
    df_temp['VitaminDlevelblood'] = pd.to_numeric(df_temp['VitaminDlevelblood'].astype(str).str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
    
    # 4. Correlation Analysis
    # Get only numeric columns
    numeric_cols = df_temp.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude UID if present
    if "UID" in numeric_cols:
        numeric_cols.remove("UID")
        
    # We want correlations between key numeric predictors and target
    # Map Target to 0/1 for correlation
    df_temp["GDM_Target"] = df_temp[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    
    key_features = [
        "AgeAtStartOfSpell", "WeightMeasured", "Height", "Body Mass Index at Booking", 
        "Parity", "Gravida", "SystolicBloodPressureCuff", "Diastolic Blood Pressure", 
        "Glucoselevel0minblood", "Glucoselevel120minblood", "GDM_Target"
    ]
    
    # Check if they exist in df_temp
    existing_key_features = [f for f in key_features if f in df_temp.columns]
    
    corr_matrix = df_temp[existing_key_features].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
    plt.title("Correlation Matrix of Key Maternal Features")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_matrix.png", dpi=150)
    plt.close()
    
    # 5. Save Text Report
    report_path = REPORTS_DIR / "eda_report.txt"
    with open(report_path, "w") as f:
        f.write("=== MATERNAL DIGITAL TWIN - EDA REPORT ===\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
        f.write(f"Target distribution (GDM):\n{target_counts.to_string()}\n\n")
        f.write("Top 20 missing value percentages:\n")
        f.write(missing_df.head(20).to_string())
        f.write("\n\nFeature Classification Statistics:\n")
        f.write(f"  Static Booking Features: {len(STATIC_FEATURES)}\n")
        f.write(f"  Dynamic Pregnancy Features: {len(DYNAMIC_FEATURES)}\n")
        f.write(f"  Delivery Outcome Features: {len(OUTCOME_FEATURES)}\n")
        
    print(f"\nEDA completed successfully. Reports and figures saved in {REPORTS_DIR}.")

if __name__ == "__main__":
    # Ensure seaborn is installed
    try:
        import seaborn as sns
    except ImportError:
        import os
        os.system("pip install seaborn")
        import seaborn as sns
    run_eda()
