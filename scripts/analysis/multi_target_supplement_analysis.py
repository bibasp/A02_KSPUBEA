"""Supporting analyses for the multi-target study (all post-processing).

Per basin, from artifacts already on disk:
  1. Day-of-year climatology EWS benchmark (1990-2010 record) with BSS,
     for BOTH walk-forward variants — the "not just seasonality" defense.
  2. Peak statistics: top-5% observed days, mean obs vs pred, underestimation.
  3. Year-by-year NSE/KGE (the "skill from the first data-scarce year" check).
  4. Drought EWS verdict per basin (consolidated from stage metrics).

Outputs:
  results/multi_target/<basin>/supplement.json
  results/multi_target/supplement_summary.csv (one row per basin)

Run: .venv/Scripts/python.exe scripts/multi_target_supplement_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydro_tl_ews.data.camels import CamelsDataset
from hydro_tl_ews.evaluation.metrics import auc_roc, brier_score, kge, nse
from gen_multi_target_configs import TARGETS
from paper_supplement_analysis import climatology_probs

RES = ROOT / "results" / "multi_target"
CLIM_PERIOD = ("1990-01-01", "2010-12-31")
LEADS = (1, 3, 7)
VARIANTS = {"wfA": "", "wfB": "_progressive"}


def per_year_skill(df: pd.DataFrame) -> dict:
    out = {}
    for yr in sorted(set(df.index.year)):
        d = df[df.index.year == yr]
        out[str(yr)] = {"NSE": float(nse(d.observed.to_numpy(), d.predicted.to_numpy())),
                        "KGE": float(kge(d.observed.to_numpy(), d.predicted.to_numpy()))}
    return out


def peak_stats(df: pd.DataFrame) -> dict:
    o, p = df.observed.to_numpy(), df.predicted.to_numpy()
    thr = np.nanquantile(o, 0.95)
    mk = o >= thr
    return {"peak_obs_mean": float(o[mk].mean()),
            "peak_pred_mean": float(p[mk].mean()),
            "peak_underest_pct": float(100 * (1 - p[mk].mean() / o[mk].mean()))}


def benchmark(hist: pd.Series, warns: pd.DataFrame, thresholds: dict) -> dict:
    out = {}
    specs = [("flood", "q95", thresholds["q95"]),
             ("flood", "q99", thresholds["q99"]),
             ("drought", "q5", thresholds["q5"])]
    for kind, pct, t in specs:
        clim = climatology_probs(hist, warns.index, t, kind)
        for L in LEADS:
            key = f"{kind}_{pct}_lead{L}d"
            y = warns[key].to_numpy()
            b_clim = float(brier_score(y, clim[f"lead{L}d"].to_numpy()))
            b_model = float(brier_score(y, warns[f"{key}_prob"].to_numpy()))
            out[key] = {
                "AUC_climatology": float(auc_roc(y, clim[f"lead{L}d"].to_numpy())),
                "AUC_model": float(auc_roc(y, warns[f"{key}_prob"].to_numpy())),
                "Brier_climatology": b_clim, "Brier_model": b_model,
                "BSS_model_vs_climatology":
                    float(1.0 - b_model / b_clim) if b_clim > 0 else float("nan"),
                "event_base_rate": float(np.mean(y)),
            }
    return out


def main() -> None:
    ds = CamelsDataset(ROOT / "data")
    rows = []
    for bid, label in TARGETS.items():
        res = RES / bid
        basin = ds.load_basin(bid)
        hist = basin.streamflow.loc[CLIM_PERIOD[0]:CLIM_PERIOD[1]].dropna()
        supp: dict = {"basin": bid, "label": label}
        row: dict = {"basin": bid, "label": label}
        for var, suffix in VARIANTS.items():
            mpath = res / f"walk_forward{suffix}_metrics.json"
            if not mpath.exists():
                continue
            m = json.loads(mpath.read_text())
            warns = pd.read_csv(res / f"walk_forward{suffix}_warnings.csv",
                                index_col=0, parse_dates=True)
            df = pd.read_parquet(res / f"walk_forward{suffix}.parquet")
            supp[var] = {
                "climatology_benchmark": benchmark(hist, warns, m["thresholds"]),
                "peaks": peak_stats(df),
                "per_year": per_year_skill(df),
                "drought_q5_lead3d": m["early_warning"].get("drought_q5_lead3d"),
            }
            b = supp[var]["climatology_benchmark"]
            row[f"{var}_flood_q95_3d_BSS"] = b["flood_q95_lead3d"]["BSS_model_vs_climatology"]
            row[f"{var}_flood_q95_3d_AUC_clim"] = b["flood_q95_lead3d"]["AUC_climatology"]
            row[f"{var}_drought_q5_3d_BSS"] = b["drought_q5_lead3d"]["BSS_model_vs_climatology"]
            row[f"{var}_peak_underest_pct"] = supp[var]["peaks"]["peak_underest_pct"]
            row[f"{var}_first_year_NSE"] = supp[var]["per_year"].get("2011", {}).get("NSE")
        (res / "supplement.json").write_text(json.dumps(supp, indent=2, default=float))
        rows.append(row)
        print(f"[{bid}] {label}: "
              f"B flood BSS {row.get('wfB_flood_q95_3d_BSS', float('nan')):.3f} "
              f"(clim AUC {row.get('wfB_flood_q95_3d_AUC_clim', float('nan')):.3f}) | "
              f"B peak underest {row.get('wfB_peak_underest_pct', float('nan')):.0f}% | "
              f"B 2011 NSE {row.get('wfB_first_year_NSE', float('nan')):.2f}")
    out = pd.DataFrame(rows).set_index("basin")
    out.to_csv(RES / "supplement_summary.csv")
    print(f"\nwrote {RES / 'supplement_summary.csv'}")


if __name__ == "__main__":
    main()
