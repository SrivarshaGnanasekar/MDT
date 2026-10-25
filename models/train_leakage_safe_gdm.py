import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, accuracy_score, brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("leakage_safe_gdm")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "maternal"
RAW_FILE = DATA_DIR / "raw" / "GDM Dataset.xlsx"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def load_and_clean_data():
    logger.info("Loading raw clinical dataset from Excel...")
    df = pd.read_excel(RAW_FILE, sheet_name="V2.0.1", header=1)
    
    # Target
    df = df.dropna(subset=["Gestational Diabetes"]).copy()
    df["Gestational Diabetes"] = df["Gestational Diabetes"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    df = df.dropna(subset=["Gestational Diabetes"]).copy()
    df["Gestational Diabetes"] = df["Gestational Diabetes"].astype(int)

    # Impute BMI from Height and Weight if missing
    if "Body Mass Index at Booking" in df.columns:
        calc_bmi = df["WeightMeasured"] / ((df["Height"] / 100.0) ** 2)
        df["Body Mass Index at Booking"] = df["Body Mass Index at Booking"].fillna(calc_bmi)
    
    # Clean numeric lab columns
    for col in ["Glucoselevelblood", "Glucoselevel0minblood", "Glucoselevel120minblood", "SystolicBloodPressureCuff", "Diastolic Blood Pressure"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace("<", "").str.strip(), errors="coerce")

    return df

def run_experiment():
    df = load_and_clean_data()
    logger.info(f"Loaded {len(df)} records. GDM prevalence: {df['Gestational Diabetes'].mean()*100:.2f}%")

    # Define Feature Sets
    early_num_features = [
        "AgeAtStartOfSpell",
        "Gestation at booking (Weeks)",
        "Body Mass Index at Booking",
        "Gravida",
        "Parity",
        "No_Of_previous_Csections",
        "Index of Multiple Deprivation Rank",
        "SystolicBloodPressureCuff",
        "Diastolic Blood Pressure"
    ]
    early_cat_features = ["Ethnicity"]
    
    glucose_features = ["Glucoselevel0minblood", "Glucoselevel120minblood"]

    # 1. Stratified Patient-Level Train/Test Split (80/20)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=RANDOM_STATE, stratify=df["Gestational Diabetes"]
    )
    logger.info(f"Split: Train={len(train_df)} patients, Test={len(test_df)} patients")

    # Pipeline Helper for Fold-level processing
    def prepare_data(tr_df, te_df, num_cols, cat_cols):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        # Fit numerical transformers on training fold only
        tr_num_imp = imputer.fit_transform(tr_df[num_cols])
        tr_num_scl = scaler.fit_transform(tr_num_imp)
        te_num_imp = imputer.transform(te_df[num_cols])
        te_num_scl = scaler.transform(te_num_imp)
        
        # Fit categorical encoder on training fold only
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        tr_cat = encoder.fit_transform(tr_df[cat_cols].astype(str))
        te_cat = encoder.transform(te_df[cat_cols].astype(str))
        
        cat_names = [f"{c}_{cat}" for c, cats in zip(cat_cols, encoder.categories_) for cat in cats]
        all_cols = num_cols + cat_names
        
        X_tr = pd.DataFrame(np.hstack([tr_num_scl, tr_cat]), columns=all_cols, index=tr_df.index)
        X_te = pd.DataFrame(np.hstack([te_num_scl, te_cat]), columns=all_cols, index=te_df.index)
        
        return X_tr, X_te, (imputer, scaler, encoder)

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Early Booking Predictors (Leakage-Safe)
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70 + "\nRUNNING EXPERIMENT 1: LEAKAGE-SAFE EARLY BOOKING PREDICTION\n" + "="*70)
    
    y_train = train_df["Gestational Diabetes"].values
    y_test = test_df["Gestational Diabetes"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    oof_lr = np.zeros(len(train_df))
    oof_rf = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))

    test_preds_lr = []
    test_preds_rf = []
    test_preds_xgb = []
    test_preds_cat = []

    fold_metrics = {"LR": [], "RF": [], "XGB": [], "CatBoost": [], "Stacking": []}

    for fold, (t_idx, v_idx) in enumerate(skf.split(train_df, y_train)):
        f_tr, f_val = train_df.iloc[t_idx], train_df.iloc[v_idx]
        y_f_tr, y_f_val = y_train[t_idx], y_train[v_idx]
        
        X_f_tr, X_f_val, _ = prepare_data(f_tr, f_val, early_num_features, early_cat_features)
        
        # Clean col names for XGBoost
        clean_cols = [c.replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_") for c in X_f_tr.columns]
        X_f_tr.columns = clean_cols
        X_f_val.columns = clean_cols
        
        # 1. Baseline: Logistic Regression
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_f_tr, y_f_tr)
        p_lr_val = lr.predict_proba(X_f_val)[:, 1]
        oof_lr[v_idx] = p_lr_val
        
        # 2. Random Forest
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X_f_tr, y_f_tr)
        p_rf_val = rf.predict_proba(X_f_val)[:, 1]
        oof_rf[v_idx] = p_rf_val
        
        # 3. XGBoost
        scale_pos = (len(y_f_tr) - sum(y_f_tr)) / sum(y_f_tr)
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
        xgb.fit(X_f_tr, y_f_tr)
        p_xgb_val = xgb.predict_proba(X_f_val)[:, 1]
        oof_xgb[v_idx] = p_xgb_val
        
        # 4. CatBoost
        cat = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.05, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0)
        cat.fit(X_f_tr, y_f_tr)
        p_cat_val = cat.predict_proba(X_f_val)[:, 1]
        oof_cat[v_idx] = p_cat_val

        fold_metrics["LR"].append(roc_auc_score(y_f_val, p_lr_val))
        fold_metrics["RF"].append(roc_auc_score(y_f_val, p_rf_val))
        fold_metrics["XGB"].append(roc_auc_score(y_f_val, p_xgb_val))
        fold_metrics["CatBoost"].append(roc_auc_score(y_f_val, p_cat_val))

    # Meta-learner on OOF predictions
    oof_meta_X = np.column_stack([oof_lr, oof_rf, oof_xgb, oof_cat])
    meta_learner = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)
    meta_learner.fit(oof_meta_X, y_train)
    oof_stacking_prob = meta_learner.predict_proba(oof_meta_X)[:, 1]
    
    # Calculate Youden's J optimal clinical screening threshold on OOF
    fpr_oof, tpr_oof, thresh_oof = roc_curve(y_train, oof_stacking_prob)
    youden_idx = np.argmax(tpr_oof - fpr_oof)
    optimal_thresh = float(thresh_oof[youden_idx])
    logger.info(f"OOF CV AUROC -> LR: {np.mean(fold_metrics['LR']):.4f}, RF: {np.mean(fold_metrics['RF']):.4f}, XGB: {np.mean(fold_metrics['XGB']):.4f}, CatBoost: {np.mean(fold_metrics['CatBoost']):.4f}")
    logger.info(f"OOF CV Stacking AUROC: {roc_auc_score(y_train, oof_stacking_prob):.4f} (Optimal Clinical Threshold: {optimal_thresh:.4f})")

    # Now Train on Full Training Set and Evaluate on Held-Out Test Set
    X_train_full, X_test_full, (imputer, scaler, encoder) = prepare_data(train_df, test_df, early_num_features, early_cat_features)
    clean_cols = [c.replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_") for c in X_train_full.columns]
    X_train_full.columns = clean_cols
    X_test_full.columns = clean_cols

    # Save processed ML files for pipeline integration
    train_ml_export = X_train_full.copy()
    train_ml_export["Gestational Diabetes"] = y_train
    test_ml_export = X_test_full.copy()
    test_ml_export["Gestational Diabetes"] = y_test
    train_ml_export.to_csv(PROCESSED_DIR / "train_ml.csv", index=False)
    test_ml_export.to_csv(PROCESSED_DIR / "test_ml.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test_clean.csv", index=False)
    logger.info("Saved updated leakage-safe train_ml.csv and test_ml.csv.")

    # Fit final models on full train
    lr_final = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
    lr_final.fit(X_train_full, y_train)
    
    rf_final = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_final.fit(X_train_full, y_train)
    
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb_final = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss")
    xgb_final.fit(X_train_full, y_train)
    
    cat_final = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.05, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0)
    cat_final.fit(X_train_full, y_train)

    # Predictions on unseen test set
    p_lr_test = lr_final.predict_proba(X_test_full)[:, 1]
    p_rf_test = rf_final.predict_proba(X_test_full)[:, 1]
    p_xgb_test = xgb_final.predict_proba(X_test_full)[:, 1]
    p_cat_test = cat_final.predict_proba(X_test_full)[:, 1]

    test_meta_X = np.column_stack([p_lr_test, p_rf_test, p_xgb_test, p_cat_test])
    p_stack_test = meta_learner.predict_proba(test_meta_X)[:, 1]

    def compute_all_metrics(y_true, y_prob, threshold=0.5):
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            "AUROC": roc_auc_score(y_true, y_prob),
            "AUPRC": average_precision_score(y_true, y_prob),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Sensitivity": recall_score(y_true, y_pred, zero_division=0),
            "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Brier Score": brier_score_loss(y_true, y_prob)
        }

    models_test_results = {
        "Logistic Regression (Baseline)": compute_all_metrics(y_test, p_lr_test),
        "Random Forest": compute_all_metrics(y_test, p_rf_test),
        "XGBoost": compute_all_metrics(y_test, p_xgb_test),
        "CatBoost": compute_all_metrics(y_test, p_cat_test),
        "Stacking Ensemble": compute_all_metrics(y_test, p_stack_test)
    }

    df_test_results = pd.DataFrame(models_test_results).T
    logger.info("\n=== TEST SET PERFORMANCE (4,698 Patients - Leakage-Safe Booking Model) ===")
    print(df_test_results.to_string())

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Diagnostic Glucose-Inclusive Model (To Document Leakage)
    # -------------------------------------------------------------------------
    logger.info("\n" + "="*70 + "\nRUNNING EXPERIMENT 2: DIAGNOSTIC GLUCOSE-INCLUSIVE BENCHMARK (LEAKAGE DEMONSTRATION)\n" + "="*70)
    
    num_with_glucose = early_num_features + glucose_features
    X_tr_glu, X_te_glu, _ = prepare_data(train_df, test_df, num_with_glucose, early_cat_features)
    X_tr_glu.columns = [c.replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_") for c in X_tr_glu.columns]
    X_te_glu.columns = [c.replace("[", "_").replace("]", "_").replace("<", "_").replace(" ", "_") for c in X_te_glu.columns]

    cat_glu = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.05, auto_class_weights="Balanced", random_state=RANDOM_STATE, verbose=0)
    cat_glu.fit(X_tr_glu, y_train)
    p_glu_test = cat_glu.predict_proba(X_te_glu)[:, 1]
    glu_metrics = compute_all_metrics(y_test, p_glu_test)
    logger.info(f"Glucose-Inclusive (Diagnostic) Model Test AUROC: {glu_metrics['AUROC']:.4f}, AUPRC: {glu_metrics['AUPRC']:.4f}, F1: {glu_metrics['F1']:.4f}")

    # -------------------------------------------------------------------------
    # GENERATE PUBLICATION-GRADE PLOTS
    # -------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. ROC Curves
    plt.figure(figsize=(9, 7))
    model_probs = {
        "Logistic Regression (Baseline)": p_lr_test,
        "Random Forest": p_rf_test,
        "XGBoost": p_xgb_test,
        "CatBoost": p_cat_test,
        "Stacking Ensemble": p_stack_test
    }
    colors = ["#7f7f7f", "#2ca02c", "#ff7f0e", "#1f77b4", "#d62728"]
    for (name, prob), col in zip(model_probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        lw = 2.5 if "Stacking" in name else 1.8
        plt.plot(fpr, tpr, label=f"{name} (AUROC = {auc:.3f})", color=col, linewidth=lw)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance Level (AUROC = 0.50)")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) - Early Booking GDM Prediction\n(Evaluated on N=4,698 Unseen Holdout Patients)", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gdm_early_prediction_roc_curves.png", dpi=300)
    plt.close()
    logger.info("Saved ROC Curves to reports/gdm_early_prediction_roc_curves.png")

    # 2. Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    prevalence = y_test.mean()
    for (name, prob), col in zip(model_probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, prob)
        auprc = average_precision_score(y_test, prob)
        lw = 2.5 if "Stacking" in name else 1.8
        plt.plot(rec, prec, label=f"{name} (AUPRC = {auprc:.3f})", color=col, linewidth=lw)
        
    plt.axhline(prevalence, color="k", linestyle="--", alpha=0.6, label=f"Baseline Prevalence ({prevalence*100:.1f}%)")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("Precision-Recall (PR) Curves - Early Booking GDM Prediction\n(High Class-Imbalance Setting: 7.9% Positive)", fontsize=13, fontweight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gdm_early_prediction_pr_curves.png", dpi=300)
    plt.close()
    logger.info("Saved PR Curves to reports/gdm_early_prediction_pr_curves.png")

    # 3. Calibration Curves (Reliability Diagram)
    plt.figure(figsize=(9, 7))
    for (name, prob), col in zip(model_probs.items(), colors):
        prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10, strategy="quantile")
        brier = brier_score_loss(y_test, prob)
        lw = 2.5 if "Stacking" in name else 1.8
        plt.plot(prob_pred, prob_true, marker="o", label=f"{name} (Brier = {brier:.4f})", color=col, linewidth=lw)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction of Positives")
    plt.title("Calibration Reliability Diagram - Early Booking GDM Models", fontsize=13, fontweight="bold")
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gdm_early_prediction_calibration.png", dpi=300)
    plt.close()
    logger.info("Saved Calibration Diagram to reports/gdm_early_prediction_calibration.png")

    # 4. Feature Importance (CatBoost on Early Booking Features)
    plt.figure(figsize=(10, 8))
    feat_imp = pd.Series(cat_final.get_feature_importance(), index=X_train_full.columns)
    top_feats = feat_imp.sort_values(ascending=False).head(15)
    
    # Prettier feature names
    name_map = {
        "Body_Mass_Index_at_Booking": "Booking BMI (kg/m²)",
        "AgeAtStartOfSpell": "Maternal Age (Years)",
        "SystolicBloodPressureCuff": "Systolic BP (mmHg)",
        "Diastolic_Blood_Pressure": "Diastolic BP (mmHg)",
        "Index_of_Multiple_Deprivation_Rank": "IMD Deprivation Rank",
        "Gestation_at_booking__Weeks_": "Gestational Age at Booking",
        "Gravida": "Gravida (Total Pregnancies)",
        "Parity": "Parity (Previous Births)",
        "No_Of_previous_Csections": "Previous C-sections"
    }
    top_clean_names = [name_map.get(f, f.replace("Ethnicity_", "Ethnicity: ")) for f in top_feats.index]
    
    palette = sns.color_palette("viridis", len(top_feats))
    sns.barplot(x=top_feats.values, y=top_clean_names, palette=palette)
    plt.xlabel("Relative Feature Importance (%)")
    plt.title("Top Clinical Predictors for Early GDM Detection\n(Gradient-Boosted Booking Model)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gdm_early_prediction_feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved Feature Importance to reports/gdm_early_prediction_feature_importance.png")

    # 5. Leakage vs. Leakage-Safe Comparison Plot
    plt.figure(figsize=(8, 6))
    comparison_df = pd.DataFrame([
        {"Model": "Early Booking (Leakage-Safe)", "AUROC": df_test_results.loc["Stacking Ensemble", "AUROC"], "AUPRC": df_test_results.loc["Stacking Ensemble", "AUPRC"]},
        {"Model": "Glucose-Inclusive (Diagnostic Leakage)", "AUROC": glu_metrics["AUROC"], "AUPRC": glu_metrics["AUPRC"]}
    ])
    melted = comparison_df.melt(id_vars="Model", value_vars=["AUROC", "AUPRC"], var_name="Metric", value_name="Score")
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=["#1f77b4", "#d62728"])
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height / 2),
                        ha="center", va="center", xytext=(0, 0), textcoords="offset points", color="white", fontweight="bold")
    plt.ylim(0, 1.15)
    plt.title("Impact of Diagnostic Target Leakage:\nEarly Booking Prediction vs. Diagnostic OGTT Inclusion", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "gdm_leakage_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved Leakage Comparison to reports/gdm_leakage_comparison.png")

    # Save summary report as text
    summary_txt = f"""=== MATERNAL GDM LEAKAGE-SAFE PREDICTION SUMMARY REPORT ===
Cohorts:
- Total Patients: {len(df)}
- Training Set: {len(train_df)} (80%)
- Test Set: {len(test_df)} (20%)
- GDM Prevalence: {df['Gestational Diabetes'].mean()*100:.2f}% ({df['Gestational Diabetes'].sum()} positive cases)

1. Test Set Performance (Early Booking Features, No OGTT Leakage):
{df_test_results.to_string()}

2. Comparison with Diagnostic Glucose-Inclusive Model:
- Early Booking Stacking AUROC: {df_test_results.loc['Stacking Ensemble', 'AUROC']:.4f} | AUPRC: {df_test_results.loc['Stacking Ensemble', 'AUPRC']:.4f}
- Glucose-Inclusive Diagnostic AUROC: {glu_metrics['AUROC']:.4f} | AUPRC: {glu_metrics['AUPRC']:.4f}
- Note: Inclusion of OGTT (0-min and 120-min) produces artificially near-perfect scores due to target definition overlap.
"""
    with open(REPORTS_DIR / "gdm_leakage_safe_report.txt", "w") as f:
        f.write(summary_txt)
    logger.info("Saved full report to reports/gdm_leakage_safe_report.txt")

    print("\n" + summary_txt)

if __name__ == "__main__":
    run_experiment()
