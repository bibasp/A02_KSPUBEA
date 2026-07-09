"""Run the full multi-target evaluation once the full pretrain has finished.

Usage:
    python scripts/run_multi_target.py            # run everything
    python scripts/run_multi_target.py 09107000   # run one basin only
    python scripts/run_multi_target.py --summary  # (re)build the summary table

Per basin the stage order is: finetune_conservative, finetune_progressive,
local_baseline, zero_shot, walk_forward (A), walk_forward_progressive (B).
Stages whose metrics/checkpoint outputs already exist are skipped, so the
script is resumable. A failing stage is logged and the run continues with the
next basin. Everything lands under results/multi_target/<basin>/, and a
cross-basin summary is written to results/multi_target/summary.csv.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = ROOT / "configs" / "multi_target"
RES_DIR = ROOT / "results" / "multi_target"
PRETRAIN_CKPT = ROOT / "results" / "checkpoints" / "pretrain.pt"
PY = sys.executable

sys.path.insert(0, str(ROOT / "scripts"))
from gen_multi_target_configs import TARGETS  # noqa: E402

STAGE_ORDER = ["finetune_conservative", "finetune_progressive",
               "local_baseline", "zero_shot",
               "walk_forward", "walk_forward_progressive"]

# stage -> output file that marks it done
DONE_MARKER = {
    "finetune_conservative": "checkpoints/finetune_conservative.pt",
    "finetune_progressive": "checkpoints/finetune_progressive.pt",
    "local_baseline": "checkpoints/local_baseline.pt",
    "zero_shot": "zero_shot_metrics.json",
    "walk_forward": "walk_forward_metrics.json",
    "walk_forward_progressive": "walk_forward_progressive_metrics.json",
}


def run_basin(bid: str) -> None:
    res = RES_DIR / bid
    res.mkdir(parents=True, exist_ok=True)
    log_path = res / "run.log"
    for stage in STAGE_ORDER:
        marker = res / DONE_MARKER[stage]
        if marker.exists():
            print(f"[{bid}] {stage}: already done, skipping")
            continue
        cfg = CFG_DIR / f"{bid}_{stage}.yaml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"{cfg} missing — run scripts/gen_multi_target_configs.py")
        print(f"[{bid}] {stage}: running ...")
        with open(log_path, "a", encoding="utf-8") as lf:
            rc = subprocess.run(
                [PY, str(ROOT / "scripts" / "run_experiment.py"),
                 "--config", str(cfg)],
                cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
            ).returncode
        if rc != 0:
            print(f"[{bid}] {stage}: FAILED (rc={rc}) — see {log_path}; "
                  f"continuing with next basin")
            return
    print(f"[{bid}] all stages complete")


def summarize() -> None:
    import pandas as pd
    rows = []
    for bid, label in TARGETS.items():
        res = RES_DIR / bid
        row = {"basin": bid, "label": label}
        zs = res / "zero_shot_metrics.json"
        if zs.exists():
            m = json.loads(zs.read_text())
            row.update(zero_shot_NSE=m.get("NSE"), zero_shot_KGE=m.get("KGE"))
        for key, prefix in (("walk_forward_metrics.json", "wfA"),
                            ("walk_forward_progressive_metrics.json", "wfB")):
            p = res / key
            if p.exists():
                m = json.loads(p.read_text())
                row[f"{prefix}_NSE"] = m["continuous"]["NSE"]
                row[f"{prefix}_KGE"] = m["continuous"]["KGE"]
                row[f"{prefix}_PBIAS"] = m["continuous"]["PBIAS"]
                ew = m.get("early_warning", {})
                q95_3d = ew.get("flood_q95_lead3d", {})
                row[f"{prefix}_flood_q95_3d_AUC"] = q95_3d.get("AUC")
                row[f"{prefix}_flood_q95_3d_Brier"] = q95_3d.get("Brier")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("basin")
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / "summary.csv"
    df.to_csv(out)
    print(df.round(3).to_string())
    print(f"\nwrote {out}")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--summary" in args:
        summarize()
        return
    if not PRETRAIN_CKPT.exists():
        sys.exit(f"{PRETRAIN_CKPT} not found — wait for the full pretrain to "
                 "finish (check results/pretrain_full.log), then rerun.")
    basins = [a for a in args if not a.startswith("-")] or list(TARGETS)
    unknown = set(basins) - set(TARGETS)
    if unknown:
        sys.exit(f"Unknown basin(s): {unknown}. Known: {list(TARGETS)}")
    for bid in basins:
        run_basin(bid)
    summarize()


if __name__ == "__main__":
    main()
