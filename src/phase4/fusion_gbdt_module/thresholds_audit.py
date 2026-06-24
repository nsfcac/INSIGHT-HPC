from __future__ import annotations

from src.phase4.fusion_gbdt_module.constants import *
import json, pickle, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression
    from sklearn.inspection import partial_dependence
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from src.phase4.fusion_gbdt import *


# Pick the threshold maximising F1 on the validation scores.
def find_threshold_best_f1(probs: np.ndarray, y: np.ndarray) -> float:
    if y.sum() == 0 or len(y) == 0:
        return 0.5
    ts = np.linspace(0.05, 0.95, 181)
    best_t, best_f = 0.5, -1.0
    for t in ts:
        p, r, f = precision_recall_at(probs, y, float(t))
        if f > best_f:
            best_f, best_t = f, float(t)
    return best_t


# Pick the highest-recall threshold that still meets a target precision.
def find_threshold_for_precision(
    probs: np.ndarray, y: np.ndarray, target_p: float, fallback: float
) -> float:
    ts = np.linspace(0.1, 0.99, 181)
    best_t, best_r = None, -1.0
    for t in ts:
        p, r, _ = precision_recall_at(probs, y, float(t))
        if p >= target_p and r > best_r:
            best_r, best_t = r, float(t)
    return best_t if best_t is not None else fallback


# Pick the highest-precision threshold that still meets a target recall.
def find_threshold_for_recall(
    probs: np.ndarray, y: np.ndarray, target_r: float, fallback: float
) -> float:
    ts = np.linspace(0.05, 0.80, 151)[::-1]
    best_t, best_p = None, -1.0
    for t in ts:
        p, r, _ = precision_recall_at(probs, y, float(t))
        if r >= target_r and p > best_p:
            best_p, best_t = p, float(t)
    return best_t if best_t is not None else fallback


# Compute point-level precision/recall/F1 at a probability threshold.
def precision_recall_at(
    probs: np.ndarray, y: np.ndarray, t: float
) -> tuple[float, float, float]:
    pred = (probs >= t).astype(np.int8)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# Compute event-level precision/recall/F1 at a threshold via GT matching.
def nab_precision_recall_at(
    val_eps_df: pd.DataFrame, gt_val: pd.DataFrame, t: float
) -> tuple[int, int, int, float, float, float]:
    from src.phase4.score_fusion import event_level_match

    pos = val_eps_df[val_eps_df["fusion_prob"] >= t]
    tp, fp, fn, _, _ = event_level_match(pos, gt_val)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return tp, fp, fn, p, r, f


# Pick the CONFIRMED-tier threshold by event-level F1 within the P/R/F1 floors.
def find_threshold_confirmed_nab(
    val_eps_df: pd.DataFrame,
    gt_val: pd.DataFrame,
    floor_p: float = 0.85,
    floor_r: float = 0.0,
    floor_f1: float = 0.0,
    incumbent_t: Optional[float] = None,
    tie_eps: float = 1e-6,
) -> tuple[float, dict]:
    ts = np.linspace(0.05, 0.95, 181)
    grid_step = float(ts[1] - ts[0])
    uncons_t, uncons_f1 = 0.5, -1.0
    trace: list[tuple[float, float, float, float]] = []
    passing: list[tuple[float, float, float, float]] = []  # (t, f1, p, r)

    for t in ts:
        _, _, _, p, r, f1 = nab_precision_recall_at(val_eps_df, gt_val, float(t))
        trace.append((float(t), p, r, f1))
        if f1 > uncons_f1:
            uncons_t, uncons_f1 = float(t), f1
        if p >= floor_p and r >= floor_r and f1 >= floor_f1:
            passing.append((float(t), f1, p, r))

    if not passing:
        return uncons_t, {
            "reason": "floors_unreachable_fell_back_to_unconstrained_argmax",
            "tied_set": [],
            "trace": trace,
        }

    best_f1 = max(c[1] for c in passing)
    plateau = sorted(c[0] for c in passing if c[1] >= best_f1 - tie_eps)

    # Rule 1: stay at incumbent if it's in the tied plateau.
    if incumbent_t is not None:
        inc = float(incumbent_t)
        for t in plateau:
            if abs(t - inc) <= grid_step + 1e-9:
                return inc, {
                    "reason": "stay_at_incumbent_tied_plateau",
                    "tied_set": plateau,
                    "trace": trace,
                }

    # Rule 2: median of plateau (mid-plateau is most noise-stable).
    mid = plateau[len(plateau) // 2]
    reason = (
        "midpoint_of_tied_plateau" if len(plateau) > 1 else "unique_max_f1_in_floors"
    )
    return mid, {"reason": reason, "tied_set": plateau, "trace": trace}


# Pick the CRITICAL-tier threshold meeting a target event-level precision.
def find_threshold_precision_nab(
    val_eps_df: pd.DataFrame, gt_val: pd.DataFrame, target_p: float, fallback: float
) -> float:
    ts = np.linspace(0.1, 0.99, 181)
    best_t, best_f1 = None, -1.0
    fallback_t, fallback_p = None, -1.0
    for t in ts:
        _, _, _, p, r, f1 = nab_precision_recall_at(val_eps_df, gt_val, float(t))
        if p > fallback_p:
            fallback_t, fallback_p = float(t), p
        if p >= target_p and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    if best_t is not None:
        return best_t
    return fallback_t if fallback_t is not None else fallback


# Pick the CANDIDATE-tier threshold meeting a target event-level recall.
def find_threshold_recall_nab(
    val_eps_df: pd.DataFrame,
    gt_val: pd.DataFrame,
    target_r: float,
    floor_p: float,
    fallback: float,
) -> float:
    ts = np.linspace(0.05, 0.80, 151)[::-1]
    best_t, best_p = None, -1.0
    for t in ts:
        _, _, _, p, r, _ = nab_precision_recall_at(val_eps_df, gt_val, float(t))
        if r >= target_r and p >= floor_p and p > best_p:
            best_p, best_t = p, float(t)
    return best_t if best_t is not None else fallback


# Read an attribute from a bundle object or dict, with a default.
def bundle_attribute(bundle, name: str, default=None):
    if bundle is None:
        return default
    if hasattr(bundle, name):
        return getattr(bundle, name)
    if hasattr(bundle, "get"):
        return bundle.get(name, default)
    return default


# Evaluate the incumbent bundle's confirmed threshold on the val set.
def incumbent_validation_metrics(
    incumbent_bundle, val_eps_df: pd.DataFrame, gt_val: pd.DataFrame
) -> Optional[tuple[float, float, float, float]]:
    if incumbent_bundle is None:
        return None
    t_old = bundle_attribute(incumbent_bundle, "t_confirmed")
    if t_old is None:
        return None
    _, _, _, p, r, f1 = nab_precision_recall_at(val_eps_df, gt_val, float(t_old))
    return float(t_old), p, r, f1


# Score episodes with the GBDT and assign fusion tiers by threshold.
def apply_fusion_gbdt(episodes: pd.DataFrame, bundle: FusionBundle) -> pd.DataFrame:
    if len(episodes) == 0:
        out = episodes.copy()
        out["fusion_prob"] = pd.Series([], dtype=np.float32)
        out["fusion_tier"] = pd.Series([], dtype=str)
        return out

    X = episodes[bundle.feature_columns].astype(np.float32).to_numpy()
    probs = bundle.clf.predict_proba(X)[:, 1].astype(np.float32)

    tier = np.full(len(probs), "NONE", dtype=object)
    tier[probs >= bundle.t_candidate] = "CANDIDATE"
    tier[probs >= bundle.t_confirmed] = "CONFIRMED"
    tier[probs >= bundle.t_critical] = "CRITICAL"

    out = episodes.copy()
    out["fusion_prob"] = probs
    out["fusion_tier"] = tier
    return out


# Pickle a FusionBundle to fusion_gbdt.pkl.
def save_bundle(bundle: FusionBundle, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fusion_gbdt.pkl"
    with open(path, "wb") as f:
        pickle.dump(
            {
                "clf": bundle.clf,
                "t_critical": bundle.t_critical,
                "t_confirmed": bundle.t_confirmed,
                "t_candidate": bundle.t_candidate,
                "feature_columns": bundle.feature_columns,
                "calibration_stats": bundle.calibration_stats,
                "training_stats": bundle.training_stats,
            },
            f,
        )
    return path


# Load a FusionBundle from a pickle file.
def load_bundle(pickle_path: Path) -> FusionBundle:
    with open(pickle_path, "rb") as f:
        d = pickle.load(f)
    return FusionBundle(
        clf=d["clf"],
        t_critical=d["t_critical"],
        t_confirmed=d["t_confirmed"],
        t_candidate=d["t_candidate"],
        feature_columns=d["feature_columns"],
        calibration_stats=d.get("calibration_stats", {}),
        training_stats=d.get("training_stats", {}),
    )


# Write feature importance, SHAP, partial-dependence, and audit JSON for the model.
def emit_auditability_outputs(
    bundle: FusionBundle,
    episodes: pd.DataFrame,
    labels: Optional[pd.Series],
    out_dir: Path,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    imp = None
    imp_std = None
    if len(episodes) > 0:
        try:
            from sklearn.inspection import permutation_importance

            X_samp = episodes[bundle.feature_columns].astype(np.float32).to_numpy()
            if labels is not None and len(labels) == len(X_samp):
                y_samp = np.asarray(labels, dtype=np.int8)
                pi = permutation_importance(
                    bundle.clf,
                    X_samp,
                    y_samp,
                    n_repeats=10,
                    random_state=42,
                    scoring="roc_auc",
                    n_jobs=1,
                )
                imp = pi.importances_mean
                imp_std = pi.importances_std
        except Exception as e:
            written["feature_importance_error"] = f"{type(e).__name__}: {e}"
    if imp is None:
        imp = np.zeros(len(bundle.feature_columns))
        imp_std = np.zeros(len(bundle.feature_columns))
    imp_df = pd.DataFrame(
        {
            "feature": bundle.feature_columns,
            "importance": imp,
            "importance_std": imp_std,
        }
    ).sort_values("importance", ascending=False)
    imp_path = out_dir / "feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    written["feature_importance"] = str(imp_path)

    if len(episodes) == 0:
        return written

    X = episodes[bundle.feature_columns].astype(np.float32).to_numpy()

    # SHAP values (per-event and summary).
    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(bundle.clf)
            shap_values = explainer.shap_values(X)
            if isinstance(
                shap_values, list
            ):  # binary classifier returns [class_0, class_1]
                shap_values = shap_values[1]
            shap_df = pd.DataFrame(
                shap_values, columns=[f"shap_{c}" for c in bundle.feature_columns]
            )
            keep_cols = [
                "hostname",
                "component",
                "episode_start",
                "episode_end",
                "fusion_prob",
                "fusion_tier",
            ]
            keep = [c for c in keep_cols if c in episodes.columns]
            shap_df = pd.concat(
                [episodes[keep].reset_index(drop=True), shap_df], axis=1
            )
            shap_parquet = out_dir / "shap_per_event.parquet"
            shap_df.to_parquet(shap_parquet, index=False)
            written["shap_per_event"] = str(shap_parquet)

            # Summary plot.
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    shap_values, X, feature_names=bundle.feature_columns, show=False
                )
                summary_path = out_dir / "shap_summary.png"
                plt.tight_layout()
                plt.savefig(summary_path, dpi=120, bbox_inches="tight")
                plt.close()
                written["shap_summary"] = str(summary_path)
            except Exception as e:
                written["shap_summary_error"] = f"{type(e).__name__}: {e}"
        except Exception as e:
            written["shap_error"] = f"{type(e).__name__}: {e}"
    else:
        written["shap_error"] = "shap not installed; pip install shap"

    # Partial dependence plots for top-5 features.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top5 = imp_df.head(5)["feature"].tolist()
        top5_idx = [bundle.feature_columns.index(f) for f in top5]
        fig, axes = plt.subplots(1, len(top5_idx), figsize=(4 * len(top5_idx), 4))
        if len(top5_idx) == 1:
            axes = [axes]
        for ax, fidx, fname in zip(axes, top5_idx, top5):
            pd_res = partial_dependence(bundle.clf, X, [fidx], kind="average")
            ax.plot(pd_res["grid_values"][0], pd_res["average"][0])
            ax.set_xlabel(fname)
            ax.set_ylabel("partial dep (log-odds)")
            ax.set_title(fname)
        pdp_path = out_dir / "partial_dependence.png"
        plt.tight_layout()
        plt.savefig(pdp_path, dpi=120, bbox_inches="tight")
        plt.close()
        written["partial_dependence"] = str(pdp_path)
    except Exception as e:
        written["pdp_error"] = f"{type(e).__name__}: {e}"

    # Summary JSON so every eval run writes a machine-readable audit.
    audit_json = out_dir / "audit_summary.json"
    with open(audit_json, "w") as f:
        json.dump(
            {
                "feature_importance": imp_df.to_dict(orient="records"),
                "calibration_stats": bundle.calibration_stats,
                "training_stats": bundle.training_stats,
                "artifacts": written,
            },
            f,
            indent=2,
            default=str,
        )
    written["audit_summary"] = str(audit_json)
    return written
