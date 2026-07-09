"""Score every multi-target local-baseline checkpoint on 2011-2014.

The local_baseline stage only saves a checkpoint + training history; this
script evaluates each basin's from-scratch model on the held-out window under
the zero-shot protocol (2009-2010 normalizer), completing the per-basin
ladder: local-from-scratch vs zero-shot vs fine-tuned vs walk-forward.

Run: python scripts/eval_local_baselines_multi_target.py
Out: results/multi_target/<basin>/local_baseline_metrics.json (+ predictions)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hydro_tl_ews.utils.config import ExperimentConfig
from stages.zero_shot_stage import run_zero_shot
from gen_multi_target_configs import TARGETS, EVAL_PERIOD, NORM_PERIOD


def main() -> None:
    rows = []
    for bid, label in TARGETS.items():
        res = ROOT / "results" / "multi_target" / bid
        ckpt = res / "checkpoints" / "local_baseline.pt"
        if not ckpt.exists():
            print(f"[{bid}] no local_baseline checkpoint — skipping")
            continue
        cfg = ExperimentConfig(
            name=f"local_baseline_eval_{label}",
            stage="zero_shot",
            data={"source": "camels", "camels_root": "data",
                  "target_basin": bid, "sequence_length": 365,
                  "full_period": EVAL_PERIOD,
                  "normalization_period": NORM_PERIOD},
            model={"pretrained_checkpoint": str(ckpt)},
            training={"batch_size": 256},
            output={"metrics_path": str(res / "local_baseline_metrics.json"),
                    "predictions_path": str(res / "local_baseline_predictions.csv")},
        )
        run_zero_shot(cfg)
        m = json.loads((res / "local_baseline_metrics.json").read_text())
        rows.append((bid, label, m["NSE"], m["KGE"], m["PBIAS"]))
        print(f"[{bid}] {label}: NSE {m['NSE']:.3f} | KGE {m['KGE']:.3f} "
              f"| PBIAS {m['PBIAS']:.1f}%")
    print("\nbasin      label                          NSE      KGE    PBIAS%")
    for bid, label, n, k, p in rows:
        print(f"{bid}  {label:28s} {n:8.3f} {k:8.3f} {p:8.1f}")


if __name__ == "__main__":
    main()
