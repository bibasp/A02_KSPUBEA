"""Generate per-target experiment configs for the multi-basin evaluation.

Writes configs/multi_target/<basin>_<stage>.yaml for every target basin in
TARGETS, all pointing at the full-CAMELS pretrain checkpoint
(results/checkpoints/pretrain.pt) and writing outputs under
results/multi_target/<basin>/.

Protocol is the corrected data-scarce protocol used for the Merced rerun on
2026-07-02: warmup 2009-01-01..2010-08-31 (train) / val targets 2010-09..12,
walk-forward 2011-2014 with refit_train_start 2009-01-01, val_tail_days 90,
90-day refit cadence, online bias correction. All seven basins were screened
for >=99% daily-flow coverage 1990-2014 and gap-free 2009-2014, so the same
dates work everywhere (see results/multi_target_screen.csv).

Regenerate anytime: python scripts/gen_multi_target_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hydro_tl_ews.utils.config import ExperimentConfig  # noqa: E402  (validation)

# basin id -> short label used in run names
TARGETS = {
    "11264500": "merced_ca_sierra_snow",
    "09107000": "taylor_co_rockies_snow",
    "14222500": "eflewis_wa_pnw_rain",
    "02128000": "little_nc_southeast_rain",
    "01544500": "kettle_pa_northeast_mixed",
    "11224500": "losgatos_ca_semiarid",
    "05507600": "lick_mo_plains",
}

PRETRAIN_CKPT = "results/checkpoints/pretrain.pt"
OUT_CFG_DIR = ROOT / "configs" / "multi_target"

WARMUP = ["2009-01-01", "2010-08-31"]
VALIDATION = ["2009-09-02", "2010-12-31"]
FULL_PERIOD = ["1990-01-01", "2014-12-31"]
EVAL_PERIOD = ["2011-01-01", "2014-12-31"]
NORM_PERIOD = ["2009-01-01", "2010-12-31"]


def _base(bid: str, label: str, stage: str) -> dict:
    return {
        "name": f"{stage}_{label}",
        "stage": stage,
        "seed": 42,
        "data": {
            "source": "camels",
            "camels_root": "data",
            "target_basin": bid,
            "sequence_length": 365,
        },
    }


def build_configs(bid: str, label: str) -> dict[str, dict]:
    res = f"results/multi_target/{bid}"
    cfgs: dict[str, dict] = {}

    for mode in ("conservative", "progressive"):
        c = _base(bid, label, f"finetune_{mode}")
        c["data"].update(warmup_period=WARMUP, validation_period=VALIDATION)
        c["model"] = {"pretrained_checkpoint": PRETRAIN_CKPT,
                      "hidden_size": 256, "dropout": 0.4}
        training = {"batch_size": 64, "patience": 3,
                    "head_lr": 1.0e-3, "weight_decay": 0.0}
        if mode == "conservative":
            training["epochs"] = 5
        else:
            training.update(epochs_head_only=5, epochs_progressive=5,
                            lstm_lr=1.0e-5, unfreeze_fraction=0.25)
        c["training"] = training
        c["output"] = {"checkpoint_path": f"{res}/checkpoints/finetune_{mode}.pt",
                       "history_path": f"{res}/history/finetune_{mode}.json"}
        cfgs[f"finetune_{mode}"] = c

    c = _base(bid, label, "local_baseline")
    c["data"].update(warmup_period=WARMUP, validation_period=VALIDATION)
    c["model"] = {"hidden_size": 256, "dropout": 0.4}
    c["training"] = {"batch_size": 64, "epochs": 50, "patience": 10,
                     "head_lr": 1.0e-3}
    c["output"] = {"checkpoint_path": f"{res}/checkpoints/local_baseline.pt",
                   "history_path": f"{res}/history/local_baseline.json"}
    cfgs["local_baseline"] = c

    c = _base(bid, label, "zero_shot")
    c["data"].update(full_period=EVAL_PERIOD, normalization_period=NORM_PERIOD)
    c["model"] = {"pretrained_checkpoint": PRETRAIN_CKPT}
    c["training"] = {"batch_size": 256}
    c["output"] = {"metrics_path": f"{res}/zero_shot_metrics.json",
                   "predictions_path": f"{res}/zero_shot_predictions.csv"}
    cfgs["zero_shot"] = c

    for approach, ft_ckpt in (
        ("conservative", f"{res}/checkpoints/finetune_conservative.pt"),
        ("progressive", f"{res}/checkpoints/finetune_progressive.pt"),
    ):
        suffix = "" if approach == "conservative" else "_progressive"
        c = _base(bid, label, "walk_forward")
        c["name"] = f"walk_forward{suffix}_{label}"
        c["data"].update(full_period=FULL_PERIOD)
        c["model"] = {"pretrained_checkpoint": ft_ckpt,
                      "hidden_size": 256, "dropout": 0.4}
        wf = {
            "initial_train_end": "2010-12-31",
            "eval_end": "2014-12-31",
            "refit_every_days": 90,
            "online_bias_correction": True,
            "val_tail_days": 90,
            "refit_train_start": "2009-01-01",
        }
        if approach == "progressive":
            wf["approach"] = "progressive"
            wf["fine_tune"] = {"head_lr": 1.0e-3, "lstm_lr": 1.0e-5,
                               "epochs_head_only": 2, "epochs_progressive": 2,
                               "patience": 2, "unfreeze_fraction": 0.25}
        else:
            wf["fine_tune"] = {"head_lr": 1.0e-3, "lstm_lr": 1.0e-5,
                               "epochs_head_only": 3, "epochs_progressive": 0,
                               "patience": 2, "unfreeze_fraction": 0.0}
        c["walk_forward"] = wf
        c["evaluation"] = {
            "lead_times": [1, 3, 7],
            "threshold_specs": [
                {"kind": "flood", "percentile": "q95"},
                {"kind": "flood", "percentile": "q99"},
                {"kind": "drought", "percentile": "q5"},
            ],
            "metrics": ["NSE", "KGE", "PBIAS", "AUC", "F1", "Brier"],
        }
        c["xai"] = {"enabled": False}  # SHAP sweep deferred; enable per basin
        c["output"] = {
            "results_path": f"{res}/walk_forward{suffix}.parquet",
            "warnings_path": f"{res}/walk_forward{suffix}_warnings.csv",
            "metrics_path": f"{res}/walk_forward{suffix}_metrics.json",
        }
        cfgs[f"walk_forward{suffix}"] = c
    return cfgs


def main() -> None:
    OUT_CFG_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for bid, label in TARGETS.items():
        for key, cfg in build_configs(bid, label).items():
            path = OUT_CFG_DIR / f"{bid}_{key}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False),
                            encoding="utf-8")
            ExperimentConfig.from_yaml(path)  # validate round-trip
            n += 1
    print(f"wrote {n} configs to {OUT_CFG_DIR}")


if __name__ == "__main__":
    main()
