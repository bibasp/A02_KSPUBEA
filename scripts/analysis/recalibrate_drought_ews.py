"""Drought-EWS probability-mapping recalibration (added 2026-07-01).

Root cause of the degenerate drought probabilities: the operational mapping
assumes Gaussian residuals with sigma = 25% of the threshold. For drought Q5
the threshold is 0.033 mm/day, so sigma ~ 0.008 mm/day — vastly tighter than
the model's actual error (~0.5-1 mm/day). Any prediction more than ~0.02 from
the threshold saturates to probability 0 or 1, producing the observed
71%-at-floor / 25%-at-one distribution and Brier scores worse than climatology.

Fix (leakage-free): estimate sigma from the fine-tuned model's residuals on
the WARMUP period only (2009-2010 — data the operational system already has),
clamp predictions at 0, and re-derive the warning probabilities on the
evaluation window. Flood metrics are recomputed with the same sigma as a
sensitivity check.

Output: results/ews_recalibrated.json
Run:    .venv/Scripts/python.exe scripts/recalibrate_drought_ews.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from torch.utils.data import DataLoader

from hydro_tl_ews.data.camels import CamelsDataset, STATIC_ATTRIBUTES
from hydro_tl_ews.data.datasets import MultiBasinSequenceDataset
from hydro_tl_ews.data.preprocessing import Normalizer, StaticNormalizer
from hydro_tl_ews.evaluation.extreme_thresholds import (
    ExtremeThresholds,
    predicted_warning_probabilities,
    warning_labels,
)
from hydro_tl_ews.evaluation.metrics import auc_roc, brier_score, f1_at_threshold
from hydro_tl_ews.training.trainer import Trainer

TARGET = "11264500"
RESULTS = ROOT / "results"
WARMUP = ("2009-01-01", "2010-12-31")
LEADS = (1, 3, 7)


def warmup_residual_sigma() -> tuple[float, int]:
    """Std of (obs - clamped pred) for the fine-tuned model on the warmup
    window — information available to the operational system before any
    evaluation data exists."""
    ds = CamelsDataset(str(ROOT / "data"))
    target = ds.load_basin(TARGET)
    attrs = ds.load_attributes()
    dyn_norm = Normalizer.fit(target.forcings.loc[WARMUP[0]:WARMUP[1]])
    static_norm = StaticNormalizer.fit(attrs.loc[:, STATIC_ATTRIBUTES])
    warm_ds = MultiBasinSequenceDataset(
        {TARGET: target}, WARMUP, dyn_norm, static_norm, sequence_length=365)
    loader = DataLoader(warm_ds, batch_size=256, shuffle=False)
    model = Trainer.load_model(str(RESULTS / "checkpoints/finetune_conservative.pt"))
    trainer = Trainer(model=model, mode="zero_shot", head_lr=1e-3)
    preds, obs = trainer.predict(loader)
    preds = np.clip(preds, 0.0, None)
    resid = obs - preds
    return float(np.std(resid)), int(len(resid))


def main() -> None:
    sigma, n_warm = warmup_residual_sigma()
    print(f"[sigma] warmup-residual sigma = {sigma:.4f} mm/day "
          f"(n={n_warm} warmup predictions)")

    wf = json.loads((RESULTS / "walk_forward_metrics.json").read_text())
    t = wf["thresholds"]
    thr = ExtremeThresholds(q5=t["q5"], q95=t["q95"], q99=t["q99"])
    old_sigma_drought = 0.25 * abs(t["q5"])

    df = pd.read_parquet(RESULTS / "walk_forward.parquet")
    obs = pd.Series(df["observed"].to_numpy(), index=df.index)
    pred = pd.Series(np.clip(df["predicted"].to_numpy(), 0.0, None),
                     index=df.index)

    out: dict = {
        "method": "Gaussian mapping with sigma from warmup-period residuals "
                  "(2009-2010, fine-tuned model, clamped preds); predictions "
                  "clamped at 0. No evaluation-period information used for "
                  "calibration.",
        "sigma_recalibrated_mm_day": sigma,
        "sigma_old_drought_mm_day": old_sigma_drought,
        "n_warmup_residuals": n_warm,
        "early_warning": {},
    }
    for kind, pct in [("drought", "q5"), ("flood", "q95"), ("flood", "q99")]:
        labels = warning_labels(obs, thr, kind=kind, percentile=pct,
                                lead_times=LEADS)
        probs = predicted_warning_probabilities(
            pred, thr, kind=kind, percentile=pct, sigma=sigma,
            lead_times=LEADS)
        for col in labels.columns:
            y = labels[col].to_numpy()
            p = probs[col].to_numpy()
            out["early_warning"][col] = {
                "AUC": float(auc_roc(y, p)),
                "F1@0.5": float(f1_at_threshold(y, p, 0.5)),
                "Brier": float(brier_score(y, p)),
            }
        if kind == "drought":
            p3 = probs["drought_q5_lead3d"]
            out["drought_prob_distribution"] = {
                "frac_at_floor(<1e-3)": float((p3 < 1e-3).mean()),
                "frac_at_one(>0.999999)": float((p3 >= 0.999999).mean()),
                "frac_intermediate": float(((p3 >= 1e-3) & (p3 < 0.999999)).mean()),
            }

    # Brier skill vs the climatology benchmark, if available
    clim_path = RESULTS / "ews_climatology_benchmark.json"
    if clim_path.exists():
        clim = json.loads(clim_path.read_text())["benchmarks"]
        for key, m in out["early_warning"].items():
            if key in clim and clim[key]["Brier_climatology"] > 0:
                m["BSS_vs_climatology"] = float(
                    1.0 - m["Brier"] / clim[key]["Brier_climatology"])

    (RESULTS / "ews_recalibrated.json").write_text(
        json.dumps(out, indent=2, default=float))
    for k, v in out["early_warning"].items():
        bss = v.get("BSS_vs_climatology")
        print(f"[recal] {k}: AUC {v['AUC']:.3f} | Brier {v['Brier']:.3f}"
              + (f" | BSS vs clim {bss:+.2f}" if bss is not None else ""))
    print(f"[recal] drought prob distribution: {out['drought_prob_distribution']}")
    print("[done] wrote results/ews_recalibrated.json")


if __name__ == "__main__":
    main()
