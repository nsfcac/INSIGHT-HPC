from __future__ import annotations

from src.phase1.lstm_autoencoder_module.constants import *
import gc, json, os, random, subprocess, sys, time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

try:
    from scipy.ndimage import binary_dilation

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from src.utils.io_utils import load_config, load_parquet, save_parquet, apply_node_limit
from src.utils.maintenance import load_maintenance_windows, apply_maintenance_mask


# Seed Python/NumPy/PyTorch and enable deterministic algorithms.
def set_deterministic_seeds(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


# Read only the support and feature columns needed for LSTM scoring.
def read_lstm_frame(path: Path, feat_cols: list) -> Optional[pd.DataFrame]:
    try:
        schema_cols = set(pq.read_schema(path).names)
        support_cols = [
            "timestamp",
            "hostname",
            "split",
            "audit_any",
            "maintenance_flag",
            "req_power_class",
        ]
        read_cols = [c for c in support_cols if c in schema_cols]
        read_cols += [c for c in feat_cols if c in schema_cols and c not in read_cols]
        if "timestamp" not in read_cols:
            return None
        return pd.read_parquet(path, engine="pyarrow", columns=read_cols)
    except Exception:
        return None


# Pick up to max_features sensor "_avg" columns plus regime_id for the LSTM.
def select_lstm_features(columns: list, max_features: int = 20) -> list:
    selected = []
    for kw in SENSOR_KEYWORDS:
        for c in columns:
            if kw in c.lower() and c.endswith("_avg") and c not in selected:
                selected.append(c)
                break
    remaining = [
        c
        for c in columns
        if c.endswith("_avg")
        and c not in selected
        and not any(x in c.lower() for x in ["audit", "coverage", "rack_audit"])
    ]
    selected.extend(remaining[: max_features - len(selected)])
    selected = selected[:max_features]
    if "regime_id" in columns and "regime_id" not in selected:
        selected.append("regime_id")
    return selected


# Slice a feature matrix into overlapping fixed-length windows.
def extract_windows(data: np.ndarray, stride: int) -> np.ndarray:
    T, F = data.shape
    if T < WINDOW:
        return np.empty((0, WINDOW, F), dtype=np.float32)
    starts = range(0, T - WINDOW + 1, stride)
    return np.stack([data[i : i + WINDOW] for i in starts], axis=0).astype(np.float32)


# Load a node's Isolation Forest scores/flags used to clean training windows.
def load_if_scores_for_node(hostname: str, cfg: dict) -> Optional[pd.DataFrame]:
    models_dir = Path(cfg["phase1"]["output_dir"])
    p = models_dir / "isolation_forest" / "scores" / f"{hostname}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, engine="pyarrow")
        df[TS] = pd.to_datetime(df[TS], utc=True, errors="coerce")
        score_col = next(
            (c for c in ["if_anomaly_score", "if_score", "score"] if c in df.columns),
            None,
        )
        flag_col = next(
            (c for c in ["if_is_anomaly", "is_anomaly"] if c in df.columns), None
        )
        if flag_col is None and score_col is None:
            return None
        keep = [TS]
        renames = {}
        if score_col:
            keep.append(score_col)
            renames[score_col] = "if_score"
        if flag_col:
            keep.append(flag_col)
            renames[flag_col] = "if_is_anomaly"
        df = df[keep].rename(columns=renames)
        if "if_score" not in df.columns:
            df["if_score"] = df["if_is_anomaly"].astype(float) * 0.5
        return df
    except Exception:
        return None


# Build the training matrix, interpolating point spikes and excluding sustained anomalies.
def build_clean_training_matrix(
    train_df: pd.DataFrame, feat_cols: list, if_scores: Optional[pd.DataFrame]
) -> tuple[np.ndarray, np.ndarray]:
    T = len(train_df)
    mat = np.zeros((T, len(feat_cols)), dtype=np.float32)
    for j, col in enumerate(feat_cols):
        if col in train_df.columns:
            v = pd.to_numeric(train_df[col], errors="coerce").fillna(0).values
            mat[:, j] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    excluded = np.zeros(T, dtype=bool)

    if if_scores is None or if_scores.empty:
        return mat, excluded

    train_ts = pd.to_datetime(train_df[TS], utc=True, errors="coerce")
    score_map = dict(zip(if_scores[TS], if_scores["if_score"].fillna(0.0)))
    scores = train_ts.map(score_map).fillna(0.0).values.astype(float)

    flag_low = scores > POINT_SPIKE_SCORE
    for i in range(1, T - 1):
        if flag_low[i] and not flag_low[i - 1] and not flag_low[i + 1]:
            for j in range(len(feat_cols)):
                mat[i, j] = (mat[i - 1, j] + mat[i + 1, j]) / 2.0
            scores[i] = 0.0

    high_flag = scores > HIGH_SCORE_THRESH
    if high_flag.any():
        if SCIPY_AVAILABLE:
            sustained = binary_dilation(high_flag, iterations=1)
        else:
            sustained = high_flag.copy()
            sustained[1:] |= high_flag[:-1]
            sustained[:-1] |= high_flag[1:]
        excluded = sustained

    return mat, excluded


# Slice windows, skipping any that overlap an excluded (anomalous) row.
def extract_windows_safe(
    mat: np.ndarray, excluded: np.ndarray, stride: int
) -> np.ndarray:
    T, F = mat.shape
    if T < WINDOW:
        return np.empty((0, WINDOW, F), dtype=np.float32)
    windows = []
    for start in range(0, T - WINDOW + 1, stride):
        if excluded[start : start + WINDOW].any():
            continue
        windows.append(mat[start : start + WINDOW])
    if not windows:
        return np.empty((0, WINDOW, F), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


# Keep windows whose reconstruction error is below the given percentile.
def filter_windows_by_recon_error(
    windows: np.ndarray,
    model,
    device: str,
    percentile: float = ITER_CLEAN_PERCENTILE_DEFAULT,
) -> np.ndarray:
    if len(windows) == 0:
        return np.ones(len(windows), dtype=bool)
    maes = score_windows(model, windows, device=device)
    threshold = float(np.percentile(maes, percentile))
    return maes <= threshold


# Build a stable key identifying a node's feature-column schema.
def feature_schema_key(feat_cols: list) -> str:
    return f"ncols{len(feat_cols)}_{abs(hash(frozenset(feat_cols))) % 100000:05d}"


# Group nodes by power tier and feature schema.
def group_nodes_by_tier_and_schema(node_meta: dict) -> dict:
    groups: dict = {}
    for hostname, meta in node_meta.items():
        schema_key = feature_schema_key(meta["feat_cols"])
        tier_key = f"T{meta['tier']}_{schema_key}"
        groups.setdefault(tier_key, []).append(hostname)
    return groups


# LSTM sequence autoencoder.
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.encoder = nn.LSTM(
            n_features,
            hidden,
            layers,
            batch_first=True,
            dropout=DROPOUT if layers > 1 else 0,
        )
        self.decoder = nn.LSTM(
            hidden,
            hidden,
            layers,
            batch_first=True,
            dropout=DROPOUT if layers > 1 else 0,
        )
        self.output = nn.Linear(hidden, n_features)

    # Encode the input sequence and decode it back to reconstruct the input.
    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        _, (h, c) = self.encoder(x)
        dec_input = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.decoder(dec_input, (h, c))
        return self.output(dec_out)


# Train the autoencoder and return it with its train-set MAE mean and std.
def train_model(
    windows: np.ndarray, n_features: int, epochs: int = EPOCHS, device: str = "cpu"
) -> tuple:
    model = LSTMAutoencoder(n_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    dataset = torch.FloatTensor(windows).to(device)
    n = len(dataset)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i : i + BATCH]
            batch = dataset[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item()) * len(idx)
        if (ep + 1) % 10 == 0:
            print(f"      ep {ep+1:3d}/{epochs}  loss={ep_loss/n:.5f}")

    model.eval()
    all_mae = []
    with torch.no_grad():
        for i in range(0, n, BATCH * 4):
            batch = dataset[i : i + BATCH * 4]
            recon = model(batch)
            mae = (recon - batch).abs().mean(dim=[1, 2])
            all_mae.extend(mae.cpu().numpy().tolist())

    train_mae = np.array(all_mae, dtype=np.float32)
    return model, float(train_mae.mean()), float(train_mae.std()) + 1e-6


# Return the per-window reconstruction MAE for a model.
def score_windows(model, windows: np.ndarray, device: str = "cpu") -> np.ndarray:
    if model is None:
        return np.zeros(len(windows), dtype=np.float32)
    model.eval()
    dataset = torch.FloatTensor(windows).to(device)
    maes = []
    with torch.no_grad():
        for i in range(0, len(dataset), BATCH * 4):
            batch = dataset[i : i + BATCH * 4]
            recon = model(batch)
            mae = (recon - batch).abs().mean(dim=[1, 2])
            maes.extend(mae.cpu().numpy().tolist())
    return np.array(maes, dtype=np.float32)


# Average overlapping window scores back onto per-timestep values.
def window_scores_to_timestep(
    window_scores: np.ndarray, T: int, stride: int = STRIDE_SCORE
) -> np.ndarray:
    ts_scores = np.zeros(T, dtype=np.float32)
    ts_counts = np.zeros(T, dtype=np.float32)
    starts = list(range(0, T - WINDOW + 1, stride))
    for w_idx, start in enumerate(starts):
        end = start + WINDOW
        ts_scores[start:end] += window_scores[w_idx]
        ts_counts[start:end] += 1.0
    ts_counts = np.where(ts_counts == 0, 1.0, ts_counts)
    return ts_scores / ts_counts


# Require a usable CUDA device, refusing to fall back to CPU.
def require_cuda() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[lstm_ae] CUDA is required for offline detection; "
            "refusing to fall back to CPU."
        )
    try:
        torch.zeros(1, device="cuda")
    except RuntimeError as e:
        raise RuntimeError(
            "[lstm_ae] CUDA is visible but unusable; " "refusing to fall back to CPU."
        ) from e
    return "cuda"


# Run one LSTM subprocess per component on the shared GPU and merge their training logs.
def run_lstm_concurrent(force: bool) -> None:
    cfg = load_config()
    feat_aligned = Path(cfg["paths"].get("features_aligned", "data/features_aligned"))
    feat_dir = feat_aligned if feat_aligned.exists() else Path(cfg["paths"]["features"])
    model_dir = Path(cfg["phase1"]["output_dir"]) / "lstm_ae"
    model_dir.mkdir(parents=True, exist_ok=True)

    comps = [
        c["name"]
        for c in cfg["components"]
        if c["name"] != "infra" and (feat_dir / c["name"]).exists()
    ]
    if len(comps) <= 1:
        os.environ.pop("INSIGHT_HPC_LSTM_CONCURRENT", None)
        return run_lstm_autoencoder(force=force)

    print(
        f"[lstm_ae] CONCURRENT mode: launching {len(comps)} per-component "
        f"subprocesses on the shared GPU -> {comps}",
        flush=True,
    )

    procs = []
    for comp in comps:
        env = dict(os.environ)
        env["INSIGHT_HPC_LSTM_ONLY_COMP"] = comp
        env["INSIGHT_HPC_LSTM_FORCE"] = "1" if force else "0"
        env.pop("INSIGHT_HPC_LSTM_CONCURRENT", None)
        p = subprocess.Popen(
            [sys.executable, "-u", "-m", "src.phase1.lstm_autoencoder"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append((comp, p))

    outputs = {}
    for comp, p in procs:
        out, _ = p.communicate()
        outputs[comp] = (p.returncode, out.decode("utf-8", "replace"))

    failed = []
    for comp in comps:
        rc, out = outputs[comp]
        print(f"\n{'='*70}\n[lstm_ae] component={comp}  (subprocess rc={rc})\n{'='*70}")
        print(out, flush=True)
        if rc != 0:
            failed.append(comp)

    merged: dict = {}
    for comp in comps:
        part = model_dir / f"training_log__{comp}.json"
        if part.exists():
            try:
                merged.update(json.loads(part.read_text()))
            except Exception:
                pass
            part.unlink()
    (model_dir / "training_log.json").write_text(json.dumps(merged, indent=2))

    if failed:
        raise RuntimeError(f"[lstm_ae] concurrent component(s) failed: {failed}")
    print(
        f"\n[lstm_ae] CONCURRENT done. Merged training log -> "
        f"{model_dir / 'training_log.json'}",
        flush=True,
    )
