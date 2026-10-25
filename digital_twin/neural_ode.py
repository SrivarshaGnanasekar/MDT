"""
========================================================================================
CONTINUOUS-TIME MATERNAL DIGITAL TWIN NEURAL ODE SIMULATION MODULE
========================================================================================
Status: Continuous-Time Physiological Simulation Engine (RK4 Numerical Integration)
Clinical Note: Routine antenatal records capture baseline measurements at booking 
(approx. Week 12). This Neural ODE provides continuous-time trajectory simulation and 
counterfactual 'what-if' visualization of maternal physiological states:
  z[0]: Fasting Blood Glucose (mmol/L)
  z[1]: Body Mass Index (kg/m^2)
  z[2]: Systolic Blood Pressure (mmHg)
across gestational weeks 12 to 40.

Integrated via 4th-order Runge-Kutta (RK4) numerical integration over continuous 
gestational time with physiological rate bounding and monotonicity regularization.

VALIDATION STATUS (2026-10-25, models/validate_neural_ode_glucose.py):
  The cohort has a single glucose value per woman (fasting OGTT, ~24-28 wk) and no
  repeated measurements, so the state dynamics are not learned from data. On held-out
  women the simulated fasting glucose at week 26 was worse than the cohort mean
  (R^2 = -2.2; initial state R^2 = +0.06 vs +0.08 for linear regression). Trajectories
  are therefore ILLUSTRATIVE ONLY. The per-week GDM "risk head" has no time-specific
  ground truth and is no longer returned by predict_trajectory(); GDM risk comes only
  from the validated booking model (models/booking_risk.py). The method itself is
  benchmarked on synthetic longitudinal data (models/benchmark_synthetic_longitudinal_ode.py).
========================================================================================
"""
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

# Core physiological numeric features for the Maternal Neural ODE (Leakage-Safe at Booking)
ODE_NUMERIC_FEATURES = [
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

# Ethnicity categories aligned with NHS St Mary's clinical cohort
ETHNICITY_CATEGORIES = [
    "AFE", "ASE", "BGD", "CAE", "CAR", "GBR", "IND", "IRL", "MEA", "NAF", "OTH", "PAK", "WEU"
]


class MaternalODEEncoder(nn.Module):
    """
    Patient-Specific Latent State Conditioning Encoder.
    Encodes standardized baseline maternal clinical biomarkers into a continuous
    conditioning embedding vector h in R^{latent_dim}.
    Includes latent dropout (p=0.1) to regularize representation learning.
    """
    def __init__(
        self,
        in_features: int = 22,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super(MaternalODEEncoder, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, in_features] Standardized maternal baseline features.
        Returns:
            h: [Batch, latent_dim] Patient conditioning embedding.
        """
        return self.net(x)


class MaternalODEFunc(nn.Module):
    """
    Derivative function network f_theta(z(t), t, h) for Continuous-Time Neural ODE.
    Computes dz(t)/dt = f_theta(z(t), t, h) with physiological rate limits.
    
    Dynamic State Variables:
      z[0]: Maternal Fasting Glucose level (mmol/L, rate ~ 0.06 mmol/L/week)
      z[1]: Maternal Body Mass Index (BMI kg/m^2, rate ~ 0.20 kg/m^2/week)
      z[2]: Maternal Systolic Blood Pressure (mmHg, rate ~ 0.40 mmHg/week)
    """
    def __init__(self, state_dim: int = 3, latent_dim: int = 32, hidden_dim: int = 64):
        super(MaternalODEFunc, self).__init__()
        self.state_dim = state_dim
        in_dim = state_dim + 1 + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
            nn.Tanh()
        )
        # Physiologically bounded rates per gestational week:
        # Glucose: max +/- 0.08 mmol/L/week
        # BMI: max +/- 0.25 kg/m^2/week
        # SBP: max +/- 0.50 mmHg/week
        self.register_buffer("rate_scale", torch.tensor([0.08, 0.25, 0.50], dtype=torch.float32))

    def forward(self, t: Union[float, torch.Tensor], z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        t_tensor = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(batch_size)
        t_normalized = (t_tensor.reshape(batch_size, 1) / 40.0)
        inp = torch.cat([z, t_normalized, h], dim=1)
        raw_rates = self.net(inp)
        scale = self.rate_scale.to(dtype=z.dtype, device=z.device)
        return raw_rates * scale


class NeuralODETwin(nn.Module):
    """
    Continuous-Time Maternal Neural ODE Digital Twin.
    Simulates continuous metabolic and hemodynamic trajectories up to Week 40
    across regular or irregularly spaced antenatal visits via Runge-Kutta 4 (RK4) integration.
    """
    def __init__(
        self,
        in_features: int = 22,
        state_dim: int = 3,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        dropout_rate: float = 0.1
    ):
        super(NeuralODETwin, self).__init__()
        if state_dim != 3:
            raise ValueError("The maternal Neural ODE state must contain glucose, BMI, and systolic BP.")

        self.in_features = in_features
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        # Latent patient conditioning encoder with dropout regularization
        self.encoder = MaternalODEEncoder(
            in_features=in_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate
        )

        # Initial continuous state projector g_phi(x, h) -> z(t0)
        self.init_proj = nn.Sequential(
            nn.Linear(in_features + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, state_dim)
        )

        # Continuous-time ODE vector field dz/dt
        self.ode_func = MaternalODEFunc(state_dim=state_dim, latent_dim=latent_dim, hidden_dim=hidden_dim)

        # Continuous longitudinal risk head
        self.risk_head = nn.Linear(state_dim + latent_dim, 1)

    def rk4_step(
        self,
        t: Union[float, torch.Tensor],
        z: torch.Tensor,
        dt: Union[float, torch.Tensor],
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        4th-Order Runge-Kutta (RK4) numerical integrator step:
        k1 = f(t, z, h)
        k2 = f(t + dt/2, z + dt/2 * k1, h)
        k3 = f(t + dt/2, z + dt/2 * k2, h)
        k4 = f(t + dt, z + dt * k3, h)
        z_{t+dt} = z_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        t_tensor = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        dt_tensor = torch.as_tensor(dt, dtype=z.dtype, device=z.device)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.expand(z.shape[0])
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.expand(z.shape[0])
        t_tensor = t_tensor.reshape(z.shape[0])
        dt_tensor = dt_tensor.reshape(z.shape[0], 1)
        t_column = t_tensor.reshape(z.shape[0], 1)

        k1 = self.ode_func(t_column, z, h)
        k2 = self.ode_func(t_column + dt_tensor * 0.5, z + (dt_tensor * 0.5) * k1, h)
        k3 = self.ode_func(t_column + dt_tensor * 0.5, z + (dt_tensor * 0.5) * k2, h)
        k4 = self.ode_func(t_column + dt_tensor, z + dt_tensor * k3, h)
        return z + (dt_tensor / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def get_initial_state(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Anchors state to physiological baselines at the patient's booking visit:
        Glucose: ~4.5-5.0 mmol/L [3.0, 15.0]
        BMI: ~25.0-27.0 kg/m2 [15.0, 50.0]
        Systolic BP: ~115-120 mmHg [90.0, 180.0]
        """
        raw = self.init_proj(torch.cat([x, h], dim=1))
        g = torch.clamp(raw[:, 0:1] + 4.8, 3.0, 15.0)
        b = torch.clamp(raw[:, 1:2] + 25.5, 15.0, 50.0)
        bp = torch.clamp(raw[:, 2:3] + 115.0, 90.0, 180.0)
        return torch.cat([g, b, bp], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t_target: Union[float, torch.Tensor],
        t_start: Union[float, torch.Tensor] = 12.0,
        steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if steps < 1:
            raise ValueError("steps must be at least 1.")
        h = self.encoder(x)
        z = self.get_initial_state(x, h)

        t_start_tensor = torch.as_tensor(t_start, dtype=z.dtype, device=z.device)
        t_target_tensor = torch.as_tensor(t_target, dtype=z.dtype, device=z.device)
        if t_start_tensor.ndim == 0:
            t_start_tensor = t_start_tensor.expand(z.shape[0])
        if t_target_tensor.ndim == 0:
            t_target_tensor = t_target_tensor.expand(z.shape[0])
        t_start_tensor = t_start_tensor.reshape(z.shape[0])
        t_target_tensor = t_target_tensor.reshape(z.shape[0])
        dt = (t_target_tensor - t_start_tensor) / steps
        t = t_start_tensor
        for _ in range(steps):
            z_next = self.rk4_step(t, z, dt, h)
            z_g = torch.clamp(z_next[:, 0:1], 2.5, 20.0)
            z_b = torch.clamp(z_next[:, 1:2], 12.0, 60.0)
            z_bp = torch.clamp(z_next[:, 2:3], 80.0, 200.0)
            z = torch.cat([z_g, z_b, z_bp], dim=1)
            t = t + dt

        risk_inp = torch.cat([z, h], dim=1)
        r_gdm = torch.sigmoid(self.risk_head(risk_inp))
        return z, r_gdm

    def solve_trajectory(
        self,
        x: torch.Tensor,
        t_start: float = 12.0,
        t_end: float = 40.0,
        steps: int = 28,
        irregular_visits: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Continuous trajectory integration supporting irregular antenatal visits.
        Returns: (maternal_state_history, r_gdm_history, t_history)
        """
        self.eval()
        with torch.no_grad():
            h = self.encoder(x)
            z = self.get_initial_state(x, h)

            if irregular_visits is not None:
                visit_times = sorted(list(set(irregular_visits)))
                t_hist = [visit_times[0]]
                z_hist = [z.clone()]

                risk_inp = torch.cat([z, h], dim=1)
                r_gdm_hist = [torch.sigmoid(self.risk_head(risk_inp)).clone()]
                curr_t = visit_times[0]
                for next_t in visit_times[1:]:
                    dt = next_t - curr_t
                    sub_steps = max(1, int(round(dt * 2)))
                    sub_dt = dt / sub_steps
                    for _ in range(sub_steps):
                        z = self.rk4_step(curr_t, z, sub_dt, h)
                        curr_t += sub_dt
                    z_g = torch.clamp(z[:, 0:1], 2.5, 20.0)
                    z_b = torch.clamp(z[:, 1:2], 12.0, 60.0)
                    z_bp = torch.clamp(z[:, 2:3], 80.0, 200.0)
                    z = torch.cat([z_g, z_b, z_bp], dim=1)

                    risk_inp = torch.cat([z, h], dim=1)
                    t_hist.append(next_t)
                    z_hist.append(z.clone())
                    r_gdm_hist.append(torch.sigmoid(self.risk_head(risk_inp)).clone())

                return (
                    torch.stack(z_hist, dim=0),
                    torch.stack(r_gdm_hist, dim=0),
                    torch.tensor(t_hist)
                )
            else:
                dt = (t_end - t_start) / steps
                t = t_start
                t_hist = [t]
                z_hist = [z.clone()]
                risk_inp = torch.cat([z, h], dim=1)
                r_gdm_hist = [torch.sigmoid(self.risk_head(risk_inp)).clone()]
                for _ in range(steps):
                    z = self.rk4_step(t, z, dt, h)
                    z_g = torch.clamp(z[:, 0:1], 2.5, 20.0)
                    z_b = torch.clamp(z[:, 1:2], 12.0, 60.0)
                    z_bp = torch.clamp(z[:, 2:3], 80.0, 200.0)
                    z = torch.cat([z_g, z_b, z_bp], dim=1)
                    t += dt

                    risk_inp = torch.cat([z, h], dim=1)
                    t_hist.append(t)
                    z_hist.append(z.clone())
                    r_gdm_hist.append(torch.sigmoid(self.risk_head(risk_inp)).clone())

                return (
                    torch.stack(z_hist, dim=0),
                    torch.stack(r_gdm_hist, dim=0),
                    torch.tensor(t_hist)
                )


def train_neural_ode(
    epochs: int = 100,
    lr: float = 0.005,
    lambda_gdm: float = 2.0,
    lambda_mono: float = 0.5,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    patience: int = 12,
    restore_best_weights: bool = True
) -> Tuple[NeuralODETwin, Dict[str, Any]]:
    """
    Trains the continuous-time maternal Neural ODE Digital Twin on booking clinical predictors.
    Leakage-safe: Encodes 9 baseline numerical vitals + 13 one-hot ethnicity categories.
    Excludes diagnostic OGTT glucose from encoder inputs to eliminate target leakage.
    Includes latent dropout (0.1), weight decay (1e-4), and early stopping (patience=12).
    Generates training records, loss convergence plots, and trajectory fidelity visualizations.
    """
    logger.info(f"Training maternal Neural ODE for continuous trajectory simulation ({epochs} epochs, lr={lr}, weight_decay={weight_decay}, patience={patience})...")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if lr <= 0:
        raise ValueError("lr must be greater than 0.")

    train_clean_path = PROCESSED_DIR / "train_clean.csv"
    if not train_clean_path.exists():
        raise FileNotFoundError(f"Clean training dataset not found at: {train_clean_path}")

    clean_df = pd.read_csv(train_clean_path)
    if "Gestational Diabetes" not in clean_df.columns:
        raise ValueError("train_clean.csv is missing 'Gestational Diabetes' target column.")

    y_gdm_series = pd.to_numeric(clean_df["Gestational Diabetes"], errors="coerce")
    valid_mask = y_gdm_series.notna()
    clean_df = clean_df.loc[valid_mask].reset_index(drop=True)
    y_gdm = y_gdm_series.loc[valid_mask].astype(np.float32).to_numpy()

    num_df = clean_df[ODE_NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
    train_indices, validation_indices = train_test_split(
        np.arange(len(y_gdm)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_gdm
    )

    medians = num_df.iloc[train_indices].median()
    num_filled = num_df.fillna(medians)
    num_arr = num_filled.to_numpy(dtype=np.float32)

    mean_vals = num_arr[train_indices].mean(axis=0)
    std_vals = num_arr[train_indices].std(axis=0) + 1e-5
    num_scaled = (num_arr - mean_vals) / std_vals

    # Encode Ethnicity (13 categories aligned with NHS cohort)
    eth_series = clean_df["Ethnicity"].astype(str).str.strip().str.upper()
    eth_onehot = np.zeros((len(clean_df), len(ETHNICITY_CATEGORIES)), dtype=np.float32)
    for c_idx, cat in enumerate(ETHNICITY_CATEGORIES):
        eth_onehot[:, c_idx] = (eth_series == cat).astype(np.float32)

    # Combined Leakage-Safe Feature Matrix: [N, 9 + 13 = 22]
    X_scaled = np.hstack([num_scaled, eth_onehot])
    feature_cols = ODE_NUMERIC_FEATURES + [f"Ethnicity_{c}" for c in ETHNICITY_CATEGORIES]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "ode_baseline_mean.npy", mean_vals)
    np.save(MODELS_DIR / "ode_baseline_std.npy", std_vals)
    with open(MODELS_DIR / "ode_features.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(MODELS_DIR / "ode_ethnicity_categories.json", "w") as f:
        json.dump(ETHNICITY_CATEGORIES, f, indent=2)

    t_book = (
        pd.to_numeric(clean_df["Gestation at booking (Weeks)"], errors="coerce")
        .fillna(12.0).clip(10.0, 24.0).to_numpy(dtype=np.float32)
    )
    if "Gestation" in clean_df.columns:
        t_deliv = (
            pd.to_numeric(clean_df["Gestation"], errors="coerce")
            .fillna(39.0).clip(34.0, 42.0).to_numpy(dtype=np.float32)
        )
    else:
        t_deliv = np.full(len(y_gdm), 39.0, dtype=np.float32)

    obs_glucose = (
        pd.to_numeric(clean_df["Glucoselevel0minblood"], errors="coerce")
        .fillna(4.5).to_numpy(dtype=np.float32)
    )
    obs_bmi = (
        pd.to_numeric(clean_df["Body Mass Index at Booking"], errors="coerce")
        .fillna(25.5).to_numpy(dtype=np.float32)
    )
    obs_sbp = (
        pd.to_numeric(clean_df["SystolicBloodPressureCuff"], errors="coerce")
        .fillna(115.0).to_numpy(dtype=np.float32)
    )

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    t_b_tensor = torch.tensor(t_book)
    t_d_tensor = torch.tensor(t_deliv)
    gdm_tensor = torch.tensor(y_gdm).unsqueeze(1)
    obs_g_tensor = torch.tensor(obs_glucose)
    obs_b_tensor = torch.tensor(obs_bmi)
    obs_sbp_tensor = torch.tensor(obs_sbp)

    model = NeuralODETwin(
        in_features=len(feature_cols),
        state_dim=3,
        latent_dim=32,
        hidden_dim=64,
        dropout_rate=0.1
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-4)

    loss_fn = nn.BCELoss()
    component_weights = {
        "gdm_bce": lambda_gdm,
        "initial_glucose_mse": 0.1,
        "initial_bmi_mse": 0.05,
        "initial_sbp_mse": 0.01,
        "bmi_monotonicity": lambda_mono
    }
    component_names = list(component_weights)

    def calculate_components(
        indices: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        bx = X_tensor[indices]
        bgdm = gdm_tensor[indices]
        tb = t_b_tensor[indices]
        td = t_d_tensor[indices]
        _, r_gdm = model(bx, t_target=td, t_start=tb, steps=6)
        loss_gdm = loss_fn(r_gdm, bgdm)

        h = model.encoder(bx)
        z_init = model.get_initial_state(bx, h)
        loss_init_g = nn.MSELoss()(z_init[:, 0], obs_g_tensor[indices])
        loss_init_b = nn.MSELoss()(z_init[:, 1], obs_b_tensor[indices])
        loss_init_sbp = nn.MSELoss()(z_init[:, 2], obs_sbp_tensor[indices])

        dz_dt = model.ode_func(tb, z_init, h)
        loss_bmi_mono = torch.mean(torch.relu(-dz_dt[:, 1]) ** 2)
        components = {
            "gdm_bce": loss_gdm,
            "initial_glucose_mse": loss_init_g,
            "initial_bmi_mse": loss_init_b,
            "initial_sbp_mse": loss_init_sbp,
            "bmi_monotonicity": loss_bmi_mono
        }
        total = sum(component_weights[name] * components[name] for name in component_names)
        return components, total, r_gdm

    def evaluate(indices: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        model.eval()
        totals = {name: 0.0 for name in component_names}
        probabilities = []
        labels = []
        sample_count = len(indices)
        with torch.no_grad():
            for start in range(0, sample_count, batch_size):
                batch_indices = torch.as_tensor(indices[start:start + batch_size], dtype=torch.long)
                components, total, batch_probabilities = calculate_components(batch_indices)
                count = len(batch_indices)
                for name, value in components.items():
                    totals[name] += value.item() * count
                totals["total"] = totals.get("total", 0.0) + total.item() * count
                probabilities.extend(batch_probabilities.cpu().reshape(-1).tolist())
                labels.extend(gdm_tensor[batch_indices].cpu().reshape(-1).tolist())

        mean_losses = {name: value / sample_count for name, value in totals.items()}
        labels_array = np.asarray(labels, dtype=np.int32)
        probabilities_array = np.asarray(probabilities, dtype=np.float64)
        predictions = probabilities_array >= 0.5
        metrics = {
            "positive_prevalence": float(labels_array.mean()),
            "precision_at_0.5": float(precision_score(labels_array, predictions, zero_division=0)),
            "recall_at_0.5": float(recall_score(labels_array, predictions, zero_division=0)),
            "average_precision": float(average_precision_score(labels_array, probabilities_array))
        }
        return mean_losses, metrics

    train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_validation_components = None
    best_validation_metrics = None
    epochs_no_improve = 0
    stopped_early = False

    history_records = []

    for epoch in range(1, epochs + 1):
        model.train()
        shuffled_indices = train_indices_tensor[torch.randperm(len(train_indices_tensor))]
        train_totals = {name: 0.0 for name in component_names}
        train_totals["total"] = 0.0
        for start in range(0, len(shuffled_indices), batch_size):
            batch_indices = shuffled_indices[start:start + batch_size]
            optimizer.zero_grad()
            components, loss, _ = calculate_components(batch_indices)
            loss.backward()
            optimizer.step()

            count = len(batch_indices)
            for name, value in components.items():
                train_totals[name] += value.item() * count
            train_totals["total"] += loss.item() * count

        train_losses = {
            name: value / len(train_indices) for name, value in train_totals.items()
        }
        validation_losses, validation_metrics = evaluate(validation_indices)
        scheduler.step(validation_losses["total"])

        record = {
            "epoch": epoch,
            "train_loss_total": train_losses["total"],
            "val_loss_total": validation_losses["total"],
            "train_gdm_bce": train_losses["gdm_bce"],
            "val_gdm_bce": validation_losses["gdm_bce"],
            "train_glucose_mse": train_losses["initial_glucose_mse"],
            "val_glucose_mse": validation_losses["initial_glucose_mse"],
            "train_bmi_mse": train_losses["initial_bmi_mse"],
            "val_bmi_mse": validation_losses["initial_bmi_mse"],
            "train_sbp_mse": train_losses["initial_sbp_mse"],
            "val_sbp_mse": validation_losses["initial_sbp_mse"],
            "val_auprc": validation_metrics["average_precision"],
            "val_precision": validation_metrics["precision_at_0.5"],
            "val_recall": validation_metrics["recall_at_0.5"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history_records.append(record)

        if validation_losses["total"] < best_validation_loss:
            best_validation_loss = validation_losses["total"]
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            best_validation_components = validation_losses
            best_validation_metrics = validation_metrics
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                f"  Epoch {epoch:2d}/{epochs} - Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {validation_losses['total']:.4f} | "
                f"Val PR-AUC: {validation_metrics['average_precision']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f}"
            )

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}: Validation loss did not improve for "
                f"{patience} consecutive epochs. Best validation loss: {best_validation_loss:.4f} at epoch {best_epoch}."
            )
            stopped_early = True
            break

    if restore_best_weights and best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"Restored best model checkpoint from epoch {best_epoch} with val loss {best_validation_loss:.4f}")

    checkpoint_path = MODELS_DIR / "neural_ode_maternal_gdm.pth"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Saved best Neural ODE checkpoint from epoch {best_epoch} to {checkpoint_path}")

    # =========================================================================
    # GENERATE PUBLICATION VISUALIZATIONS & RECORDS
    # =========================================================================
    df_history = pd.DataFrame(history_records)
    
    # 1. Training Loss Curves Figure
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel A: Overall Total Loss Convergence
    axes[0].plot(df_history["epoch"], df_history["train_loss_total"], label="Train Loss", color="#2563EB", lw=2.2)
    axes[0].plot(df_history["epoch"], df_history["val_loss_total"], label="Val Loss", color="#DC2626", lw=2.2)
    axes[0].scatter([best_epoch], [best_validation_loss], color="#16A34A", s=110, zorder=6, label=f"Best Checkpoint (Ep {best_epoch})")
    axes[0].set_xlabel("Epoch", fontweight="bold")
    axes[0].set_ylabel("Total Loss", fontweight="bold")
    axes[0].set_title(f"Total Loss Convergence across {epochs} Epochs\n(Best Val Loss: {best_validation_loss:.4f})", fontweight="bold", fontsize=11)
    axes[0].legend(loc="upper right", frameon=True)

    # Panel B: Decomposed Validation Loss Components
    axes[1].plot(df_history["epoch"], df_history["val_gdm_bce"], label="GDM Risk BCE", color="#7C3AED", lw=2)
    axes[1].plot(df_history["epoch"], df_history["val_glucose_mse"], label="Initial Glucose MSE", color="#059669", lw=1.8)
    axes[1].plot(df_history["epoch"], df_history["val_bmi_mse"], label="Initial BMI MSE", color="#D97706", lw=1.8)
    axes[1].plot(df_history["epoch"], df_history["val_sbp_mse"], label="Initial SBP MSE", color="#475569", lw=1.8)
    axes[1].set_xlabel("Epoch", fontweight="bold")
    axes[1].set_ylabel("Component Loss", fontweight="bold")
    axes[1].set_title("Validation Loss Decomposed Components", fontweight="bold", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)

    # Panel C: Validation Average Precision (PR-AUC)
    axes[2].plot(df_history["epoch"], df_history["val_auprc"], label="Validation PR-AUC", color="#EC4899", lw=2.2)
    best_pr = df_history.loc[df_history["epoch"] == best_epoch, "val_auprc"].values[0]
    axes[2].scatter([best_epoch], [best_pr], color="#16A34A", s=110, zorder=6, label=f"At Best Ep ({best_pr:.4f})")
    axes[2].set_xlabel("Epoch", fontweight="bold")
    axes[2].set_ylabel("PR-AUC (AUPRC)", fontweight="bold")
    axes[2].set_title("Longitudinal Risk Discrimination (PR-AUC)", fontweight="bold", fontsize=11)
    axes[2].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    loss_curve_path = REPORTS_DIR / "neural_ode_training_loss_curves.png"
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    logger.info(f"Saved training loss convergence curves to {loss_curve_path}")

    # 2. Continuous Trajectory Simulation Figure (Normoglycemic vs GDM Cohorts)
    model.eval()
    val_x = X_tensor[validation_indices]
    val_y = y_gdm[validation_indices]

    normo_idx = np.where(val_y == 0)[0][:100]
    gdm_idx = np.where(val_y == 1)[0][:100]

    timeline_w = np.linspace(12.0, 40.0, 29)
    with torch.no_grad():
        z_normo, r_normo, t_w = model.solve_trajectory(val_x[normo_idx], t_start=12.0, t_end=40.0, steps=28)
        z_gdm, r_gdm, _ = model.solve_trajectory(val_x[gdm_idx], t_start=12.0, t_end=40.0, steps=28)

    # Convert to numpy [Time, Batch, State] -> [Batch, Time]
    g_normo = z_normo[:, :, 0].cpu().numpy().T
    b_normo = z_normo[:, :, 1].cpu().numpy().T
    sbp_normo = z_normo[:, :, 2].cpu().numpy().T
    r_normo = r_normo[:, :, 0].cpu().numpy().T

    g_gdm = z_gdm[:, :, 0].cpu().numpy().T
    b_gdm = z_gdm[:, :, 1].cpu().numpy().T
    sbp_gdm = z_gdm[:, :, 2].cpu().numpy().T
    r_gdm = r_gdm[:, :, 0].cpu().numpy().T

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Fasting Glucose
    axes[0, 0].plot(timeline_w, g_normo.mean(axis=0), color="#2563EB", lw=2.5, label="Normoglycemic Cohort (Mean)")
    axes[0, 0].fill_between(timeline_w, g_normo.mean(axis=0) - g_normo.std(axis=0), g_normo.mean(axis=0) + g_normo.std(axis=0), color="#2563EB", alpha=0.15)
    axes[0, 0].plot(timeline_w, g_gdm.mean(axis=0), color="#DC2626", lw=2.5, label="High-Risk GDM Cohort (Mean)")
    axes[0, 0].fill_between(timeline_w, g_gdm.mean(axis=0) - g_gdm.std(axis=0), g_gdm.mean(axis=0) + g_gdm.std(axis=0), color="#DC2626", alpha=0.15)
    axes[0, 0].set_title("Maternal Fasting Glucose Trajectory", fontweight="bold")
    axes[0, 0].set_xlabel("Gestational Age (Weeks)", fontweight="bold")
    axes[0, 0].set_ylabel("Fasting Glucose (mmol/L)", fontweight="bold")
    axes[0, 0].legend(loc="upper left", frameon=True)

    # Maternal BMI
    axes[0, 1].plot(timeline_w, b_normo.mean(axis=0), color="#2563EB", lw=2.5, label="Normoglycemic Cohort (Mean)")
    axes[0, 1].fill_between(timeline_w, b_normo.mean(axis=0) - b_normo.std(axis=0), b_normo.mean(axis=0) + b_normo.std(axis=0), color="#2563EB", alpha=0.15)
    axes[0, 1].plot(timeline_w, b_gdm.mean(axis=0), color="#DC2626", lw=2.5, label="High-Risk GDM Cohort (Mean)")
    axes[0, 1].fill_between(timeline_w, b_gdm.mean(axis=0) - b_gdm.std(axis=0), b_gdm.mean(axis=0) + b_gdm.std(axis=0), color="#DC2626", alpha=0.15)
    axes[0, 1].set_title("Maternal BMI Progression (Continuous GWG)", fontweight="bold")
    axes[0, 1].set_xlabel("Gestational Age (Weeks)", fontweight="bold")
    axes[0, 1].set_ylabel("Body Mass Index (kg/m²)", fontweight="bold")
    axes[0, 1].legend(loc="upper left", frameon=True)

    # Systolic Blood Pressure
    axes[1, 0].plot(timeline_w, sbp_normo.mean(axis=0), color="#2563EB", lw=2.5, label="Normoglycemic Cohort (Mean)")
    axes[1, 0].fill_between(timeline_w, sbp_normo.mean(axis=0) - sbp_normo.std(axis=0), sbp_normo.mean(axis=0) + sbp_normo.std(axis=0), color="#2563EB", alpha=0.15)
    axes[1, 0].plot(timeline_w, sbp_gdm.mean(axis=0), color="#DC2626", lw=2.5, label="High-Risk GDM Cohort (Mean)")
    axes[1, 0].fill_between(timeline_w, sbp_gdm.mean(axis=0) - sbp_gdm.std(axis=0), sbp_gdm.mean(axis=0) + sbp_gdm.std(axis=0), color="#DC2626", alpha=0.15)
    axes[1, 0].set_title("Systolic Blood Pressure Trajectory", fontweight="bold")
    axes[1, 0].set_xlabel("Gestational Age (Weeks)", fontweight="bold")
    axes[1, 0].set_ylabel("Systolic BP (mmHg)", fontweight="bold")
    axes[1, 0].legend(loc="upper left", frameon=True)

    # Cumulative GDM Risk Evolution
    axes[1, 1].plot(timeline_w, r_normo.mean(axis=0) * 100.0, color="#2563EB", lw=2.5, label="Normoglycemic Cohort (Mean)")
    axes[1, 1].plot(timeline_w, r_gdm.mean(axis=0) * 100.0, color="#DC2626", lw=2.5, label="High-Risk GDM Cohort (Mean)")
    axes[1, 1].set_title("Continuous Model-Estimated GDM Risk P(t)", fontweight="bold")
    axes[1, 1].set_xlabel("Gestational Age (Weeks)", fontweight="bold")
    axes[1, 1].set_ylabel("Estimated Risk (%)", fontweight="bold")
    axes[1, 1].legend(loc="upper left", frameon=True)

    plt.tight_layout()
    traj_path = REPORTS_DIR / "neural_ode_trajectory_simulation.png"
    plt.savefig(traj_path, dpi=300)
    plt.close()
    logger.info(f"Saved continuous trajectory simulation figure to {traj_path}")

    # 3. Save Detailed Text & JSON Records
    start_loss = df_history.iloc[0]["train_loss_total"]
    final_loss = df_history.iloc[-1]["train_loss_total"]
    start_val = df_history.iloc[0]["val_loss_total"]
    final_val = df_history.iloc[-1]["val_loss_total"]

    summary_text = f"""=== CONTINUOUS-TIME MATERNAL NEURAL ODE TRAINING RECORD ===
Architecture:
- Type: Patient-Conditioned Continuous-Time Latent Neural ODE (RK4 numerical integration)
- Input Features: {len(feature_cols)} baseline clinical predictors ({', '.join(feature_cols)})
- Latent Conditioning Dim: 32 | Hidden Dim: 64 | State Dim: 3 (Glucose, BMI, Systolic BP)
- Physiological Bounding: Tanh rate-scaled [0.08 mmol/L/wk, 0.25 kg/m²/wk, 0.50 mmHg/wk]
- Optimizer: Adam (Initial LR = {lr}) with ReduceLROnPlateau scheduler

Training Configuration:
- Dataset: NHS St Mary's cohort (train_clean.csv, N=18,792 patients)
- Split: 15,033 train (80%) / 3,759 validation (20%) (stratified by GDM outcome)
- Epochs: {epochs} | Batch Size: {batch_size}

Loss Convergence Analysis:
- Initial Training Loss (Epoch 1):  {start_loss:.4f}
- Final Training Loss (Epoch {epochs}): {final_loss:.4f} (Reduction: {((start_loss - final_loss) / start_loss)*100:.1f}%)
- Initial Validation Loss (Epoch 1): {start_val:.4f}
- Final Validation Loss (Epoch {epochs}): {final_val:.4f}
- Best Validation Checkpoint: Epoch {best_epoch} with Validation Combined Loss = {best_validation_loss:.4f}

Best Checkpoint Decomposed Validation Metrics:
- GDM Risk BCE Loss:           {best_validation_components['gdm_bce']:.4f}
- Baseline Glucose MSE:         {best_validation_components['initial_glucose_mse']:.4f}
- Baseline BMI MSE:             {best_validation_components['initial_bmi_mse']:.4f}
- Baseline Systolic BP MSE:     {best_validation_components['initial_sbp_mse']:.4f}
- BMI Monotonicity Penalty:     {best_validation_components['bmi_monotonicity']:.6f}
- Validation PR-AUC (AUPRC):    {best_validation_metrics['average_precision']:.4f}
- Validation Precision (at 0.5):{best_validation_metrics['precision_at_0.5']:.4f}
- Validation Recall (at 0.5):   {best_validation_metrics['recall_at_0.5']:.4f}

Generated Artifacts:
- Model Checkpoint:  models/neural_ode_maternal_gdm.pth
- Mean/Std Scalers:  models/ode_baseline_mean.npy, models/ode_baseline_std.npy
- Feature Schema:    models/ode_features.json
- Loss Curves Plot:  reports/neural_ode_training_loss_curves.png
- Trajectory Plot:   reports/neural_ode_trajectory_simulation.png
- Training History:  reports/neural_ode_training_record.json
"""
    record_txt_path = REPORTS_DIR / "neural_ode_training_record.txt"
    with open(record_txt_path, "w") as f:
        f.write(summary_text)

    record_json_path = REPORTS_DIR / "neural_ode_training_record.json"
    with open(record_json_path, "w") as f:
        json.dump({
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_components": best_validation_components,
            "best_validation_metrics": best_validation_metrics,
            "history": history_records
        }, f, indent=2)

    logger.info(f"Saved training record summary to {record_txt_path}")

    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "history": history_records,
        "medians": {col: float(medians[col]) for col in ODE_NUMERIC_FEATURES}
    }


class FetalMaternalODEPredictor:
    """
    Exploratory simulation interface for continuous maternal trajectories (glucose, BMI, systolic BP).
    Generates numerical RK4 continuous-time curves anchored to baseline booking vitals for 
    exploratory simulation and 'what-if' visualization in the clinical digital twin prototype.
    """
    def __init__(self):
        self.model_path = MODELS_DIR / "neural_ode_maternal_gdm.pth"
        self.mean_path = MODELS_DIR / "ode_baseline_mean.npy"
        self.std_path = MODELS_DIR / "ode_baseline_std.npy"
        self.num_features = ODE_NUMERIC_FEATURES
        self.ethnicity_categories = ETHNICITY_CATEGORIES
        self.feature_cols = ODE_NUMERIC_FEATURES + [f"Ethnicity_{c}" for c in ETHNICITY_CATEGORIES]
        self.model = NeuralODETwin(
            in_features=len(self.feature_cols),
            state_dim=3,
            latent_dim=32,
            hidden_dim=64,
            dropout_rate=0.0
        )
        self.is_loaded = False
        self.mean = None
        self.std = None
        self.load_model()

    def load_model(self):
        if self.model_path.exists() and self.mean_path.exists() and self.std_path.exists():
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                state_dict = torch.load(self.model_path, map_location=device)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.mean = np.load(self.mean_path)
                self.std = np.load(self.std_path)
                self.is_loaded = True
                logger.info("Continuous Maternal Neural ODE weights and scalers loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading Neural ODE weights: {e}")
                self.is_loaded = False
        else:
            logger.warning("Neural ODE checkpoint or scalers not found. Training quick baseline on first inference.")
            self.is_loaded = False

    @staticmethod
    def _anchor_trajectory_risk(
        trajectory: pd.DataFrame,
        anchor_week: float,
        target_probability: float
    ) -> pd.DataFrame:
        if trajectory.empty or not np.isfinite(target_probability):
            return trajectory

        week_column = "gestational_age_weeks" if "gestational_age_weeks" in trajectory.columns else "gestational_week"
        risk_column = "gdm_risk_probability"
        if week_column not in trajectory.columns or risk_column not in trajectory.columns:
            return trajectory

        epsilon = 1e-6
        probabilities = np.clip(trajectory[risk_column].to_numpy(dtype=float), epsilon, 1.0 - epsilon)
        anchor_index = np.abs(trajectory[week_column].to_numpy(dtype=float) - float(anchor_week)).argmin()
        logits = np.log(probabilities) - np.log1p(-probabilities)
        target = float(np.clip(target_probability, epsilon, 1.0 - epsilon))
        target_logit = np.log(target) - np.log1p(-target)

        anchored = trajectory.copy()
        anchored[risk_column] = 1.0 / (1.0 + np.exp(-(logits + target_logit - logits[anchor_index])))
        return anchored

    def extract_features(self, profile: Dict[str, Any]) -> np.ndarray:
        """Safely extracts and standardizes the 9 numeric + 13 ethnicity one-hot ODE features."""
        num_feats = []
        for feat in self.num_features:
            val = profile.get(feat, None)
            if val is None:
                if feat == "AgeAtStartOfSpell":
                    val = profile.get("Age", profile.get("maternal_age", 31.0))
                elif feat == "Gestation at booking (Weeks)":
                    val = profile.get("Booking GA", profile.get("gestational_age_weeks", 12.0))
                elif feat == "Body Mass Index at Booking":
                    val = profile.get("BMI", profile.get("bmi", 26.0))
                elif feat == "Gravida":
                    val = profile.get("gravida", 1.0)
                elif feat == "Parity":
                    val = profile.get("parity", 0.0)
                elif feat == "No_Of_previous_Csections":
                    val = profile.get("csections", 0.0)
                elif feat == "Index of Multiple Deprivation Rank":
                    val = profile.get("imdr", 13000.0)
                elif feat == "SystolicBloodPressureCuff":
                    val = profile.get("Systolic BP", profile.get("systolic_bp", 115.0))
                elif feat == "Diastolic Blood Pressure":
                    val = profile.get("Diastolic BP", profile.get("diastolic_bp", 70.0))
                else:
                    val = 0.0
            num_feats.append(float(val))

        raw_num = np.array(num_feats, dtype=np.float32)
        if self.mean is not None and self.std is not None and len(self.mean) == len(raw_num):
            scaled_num = (raw_num - self.mean) / self.std
        else:
            scaled_num = raw_num

        # One-hot encode Ethnicity (13 categories aligned with cohort)
        raw_eth = str(profile.get("Ethnicity", profile.get("ethnicity", "WEU"))).strip().upper()
        eth_vec = np.zeros(len(self.ethnicity_categories), dtype=np.float32)
        if raw_eth in self.ethnicity_categories:
            eth_vec[self.ethnicity_categories.index(raw_eth)] = 1.0
        else:
            weu_idx = self.ethnicity_categories.index("WEU")
            eth_vec[weu_idx] = 1.0

        return np.concatenate([scaled_num, eth_vec])

    def predict_trajectory(
        self,
        maternal_profile: Dict[str, Any],
        anchor_probability: Optional[float] = None,
        irregular_visits: Optional[List[float]] = None,
        vitals_history: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Solves continuous maternal trajectories across gestational weeks 12 to 40.
        Illustrative only (see module docstring). ``anchor_probability`` is accepted for
        backward compatibility and ignored; no GDM risk column is returned.
        If vitals_history is provided, incorporates recorded clinical visits as observed
        checkpoints and forward-projects from the latest observation up to Week 40.
        """
        if not self.is_loaded:
            try:
                model, _ = train_neural_ode(epochs=30)
                self.model = model
                self.mean = np.load(self.mean_path)
                self.std = np.load(self.std_path)
                self.is_loaded = True
            except Exception as e:
                logger.error(f"Neural ODE training fallback failed: {e}")
                weeks = np.array(irregular_visits) if irregular_visits else np.arange(12, 41)
                bmi_val = float(maternal_profile.get("Body Mass Index at Booking", maternal_profile.get("BMI", 26.0)))
                glu_val = float(maternal_profile.get("Glucoselevel0minblood", maternal_profile.get("Glucoselevelblood", 4.8)))
                sbp_val = float(maternal_profile.get("SystolicBloodPressureCuff", maternal_profile.get("Systolic BP", 118.0)))
                return pd.DataFrame({
                    "gestational_week": weeks,
                    "glucose_level": glu_val + 0.03 * (weeks - 12),
                    "bmi": bmi_val + 0.12 * (weeks - 12),
                    "systolic_bp": sbp_val + 0.15 * (weeks - 12),
                    "is_observed_visit": False
                })

        valid_visits = {}
        if vitals_history:
            for v in vitals_history:
                ga = v.get("gestational_age_weeks")
                if ga is not None and np.isfinite(ga) and 10.0 <= float(ga) <= 42.0:
                    ga_key = round(float(ga), 1)
                    valid_visits[ga_key] = {
                        "ga": ga_key,
                        "glucose": float(v.get("glucose_0min", v.get("glucose_blood", 4.8))),
                        "bmi": float(v.get("bmi", 26.0)) if v.get("bmi") else None,
                        "sbp": float(v.get("systolic_bp", 118.0)) if v.get("systolic_bp") else None
                    }

        if valid_visits:
            hist_gas = sorted(list(valid_visits.keys()))
            latest_ga = hist_gas[-1]
            future_gas = [float(w) for w in range(int(np.ceil(latest_ga)) + 1, 41)]
            timeline = sorted(list(set([12.0] + hist_gas + future_gas)))
        elif irregular_visits:
            timeline = sorted(list(set(irregular_visits)))
        else:
            timeline = [float(w) for w in range(12, 41)]

        profile = dict(maternal_profile)
        if valid_visits:
            latest_v = valid_visits[sorted(list(valid_visits.keys()))[-1]]
            profile["Gestation at booking (Weeks)"] = latest_v["ga"]
            if latest_v["glucose"] is not None:
                profile["Glucoselevel0minblood"] = latest_v["glucose"]
            if latest_v["bmi"] is not None:
                profile["Body Mass Index at Booking"] = latest_v["bmi"]
            if latest_v["sbp"] is not None:
                profile["SystolicBloodPressureCuff"] = latest_v["sbp"]

        scaled_features = self.extract_features(profile)
        feats_t = torch.tensor(scaled_features).unsqueeze(0)

        z_hist, r_gdm_hist, t_hist = self.model.solve_trajectory(
            feats_t,
            t_start=12.0,
            t_end=40.0,
            steps=len(timeline) - 1,
            irregular_visits=timeline
        )

        weeks = t_hist.numpy()
        states = z_hist.squeeze(1).numpy()
        risks_gdm = r_gdm_hist.squeeze(1).numpy()

        df_traj = pd.DataFrame({
            "gestational_week": weeks,
            "glucose_level": states[:, 0],
            "bmi": states[:, 1],
            "systolic_bp": states[:, 2],
            "gdm_risk_probability": risks_gdm[:, 0],
            "is_observed_visit": False
        })

        if valid_visits:
            for ga_obs, obs_data in valid_visits.items():
                idx = int(np.abs(df_traj["gestational_week"].values - ga_obs).argmin())
                df_traj.loc[idx, "is_observed_visit"] = True
                if obs_data["glucose"] is not None:
                    df_traj.loc[idx, "glucose_level"] = obs_data["glucose"]
                if obs_data["bmi"] is not None:
                    df_traj.loc[idx, "bmi"] = obs_data["bmi"]
                if obs_data["sbp"] is not None:
                    df_traj.loc[idx, "systolic_bp"] = obs_data["sbp"]

            latest_ga = sorted(list(valid_visits.keys()))[-1]
            idx_last = int(np.abs(df_traj["gestational_week"].values - latest_ga).argmin())
            shift_g = (valid_visits[latest_ga]["glucose"] - states[idx_last, 0]) if valid_visits[latest_ga]["glucose"] else 0.0
            shift_b = (valid_visits[latest_ga]["bmi"] - states[idx_last, 1]) if valid_visits[latest_ga]["bmi"] else 0.0
            shift_sbp = (valid_visits[latest_ga]["sbp"] - states[idx_last, 2]) if valid_visits[latest_ga]["sbp"] else 0.0

            future_mask = df_traj["gestational_week"] >= latest_ga
            df_traj.loc[future_mask, "glucose_level"] = np.clip(states[future_mask, 0] + shift_g, 2.5, 20.0)
            df_traj.loc[future_mask, "bmi"] = np.clip(states[future_mask, 1] + shift_b, 12.0, 60.0)
            df_traj.loc[future_mask, "systolic_bp"] = np.clip(states[future_mask, 2] + shift_sbp, 80.0, 200.0)
        else:
            booking_ga = float(maternal_profile.get("Gestation at booking (Weeks)", maternal_profile.get("Booking GA", 12.0)))
            idx_book = int(np.abs(df_traj["gestational_week"].values - booking_ga).argmin())

            obs_glu = maternal_profile.get("Glucoselevel0minblood", maternal_profile.get("Glucoselevelblood", maternal_profile.get("Fasting Glucose", None)))
            if obs_glu is not None:
                df_traj["glucose_level"] += float(obs_glu) - df_traj.loc[idx_book, "glucose_level"]

            obs_bmi = maternal_profile.get("Body Mass Index at Booking", maternal_profile.get("BMI", None))
            if obs_bmi is not None:
                df_traj["bmi"] += float(obs_bmi) - df_traj.loc[idx_book, "bmi"]

            obs_sbp = maternal_profile.get("SystolicBloodPressureCuff", maternal_profile.get("Systolic BP", None))
            if obs_sbp is not None:
                df_traj["systolic_bp"] += float(obs_sbp) - df_traj.loc[idx_book, "systolic_bp"]

        df_traj["glucose_level"] = np.clip(df_traj["glucose_level"], 2.5, 20.0)
        df_traj["bmi"] = np.clip(df_traj["bmi"], 12.0, 60.0)
        df_traj["systolic_bp"] = np.clip(df_traj["systolic_bp"], 80.0, 200.0)

        anchor_ga = float(sorted(list(valid_visits.keys()))[-1]) if valid_visits else float(maternal_profile.get("Gestation at booking (Weeks)", maternal_profile.get("Booking GA", 12.0)))
        # The per-week GDM risk head is not validated (no time-specific outcome exists): drop it.
        return df_traj.drop(columns=["gdm_risk_probability"], errors="ignore")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    model, info = train_neural_ode(epochs=100, lr=0.005, weight_decay=1e-4, patience=12, restore_best_weights=True)
    predictor = FetalMaternalODEPredictor()
    sample_profile = {
        "AgeAtStartOfSpell": 33,
        "Gestation at booking (Weeks)": 12.0,
        "Body Mass Index at Booking": 28.5,
        "Gravida": 2.0,
        "Parity": 1.0,
        "No_Of_previous_Csections": 0.0,
        "Index of Multiple Deprivation Rank": 11500.0,
        "SystolicBloodPressureCuff": 122.0,
        "Diastolic Blood Pressure": 78.0,
        "Ethnicity": "WEU",
        "Glucoselevel0minblood": 4.8
    }
    df_test = predictor.predict_trajectory(sample_profile)
    logger.info(f"Generated test trajectory across {len(df_test)} gestational timepoints:\n{df_test.head(5)}")
    predictor = FetalMaternalODEPredictor()
    sample_profile = {
        "AgeAtStartOfSpell": 33,
        "Gestation at booking (Weeks)": 14.0,
        "Body Mass Index at Booking": 29.5,
        "Gravida": 2.0,
        "Parity": 1.0,
        "No_Of_previous_Csections": 0.0,
        "Index of Multiple Deprivation Rank": 12450.0,
        "SystolicBloodPressureCuff": 126.0,
        "Diastolic Blood Pressure": 82.0,
        "Glucoselevel0minblood": 5.4
    }
    traj = predictor.predict_trajectory(sample_profile)
    print("\nSolved Continuous Trajectory (Regular Grid):\n", traj.head())

    traj_irreg = predictor.predict_trajectory(sample_profile, irregular_visits=[12.0, 16.5, 20.0, 26.0, 32.0, 38.0, 40.0])
    print("\nSolved Trajectory (Irregular Antenatal Visits):\n", traj_irreg)
