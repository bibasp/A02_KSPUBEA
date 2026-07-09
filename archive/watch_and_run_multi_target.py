"""Wait for the full pretrain to finish, then launch the multi-target study.

Runs detached (see PROJECT_STATUS_AND_CONTINUATION.txt). Polls the pretrain
log every 10 minutes; when training has ended it starts
scripts/run_multi_target.py, whose output goes to
results/multi_target/orchestrator.log.

Completion is detected by the "Pre-training complete" log line. If the
training process disappears WITHOUT that line (crash / manual kill), we still
proceed after a grace period: the checkpoint on disk is always the
best-validation snapshot, so it is usable — but the event is logged loudly so
a human reviews the pretrain log before trusting the run.
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_LOG = ROOT / "results" / "pretrain_full.log"
CHECKPOINT = ROOT / "results" / "checkpoints" / "pretrain.pt"
WATCH_LOG = ROOT / "results" / "multi_target" / "watcher.log"
ORCH_LOG = ROOT / "results" / "multi_target" / "orchestrator.log"
POLL_SECONDS = 600
STALE_HOURS = 4  # no log growth for this long => assume training ended


def note(msg: str) -> None:
    WATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    with open(WATCH_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main() -> None:
    note(f"watcher started (poll {POLL_SECONDS}s, stale after {STALE_HOURS}h)")
    while True:
        text = PRETRAIN_LOG.read_text(encoding="utf-8", errors="replace") \
            if PRETRAIN_LOG.exists() else ""
        if "Pre-training complete" in text:
            note("pretrain COMPLETE line found — launching multi-target study")
            break
        age_h = (time.time() - PRETRAIN_LOG.stat().st_mtime) / 3600 \
            if PRETRAIN_LOG.exists() else 0.0
        if age_h > STALE_HOURS:
            note(f"WARNING: pretrain log stale for {age_h:.1f} h without a "
                 "completion line — assuming training ended abnormally. "
                 "Proceeding with the best-so-far checkpoint; REVIEW "
                 "results/pretrain_full.log before trusting results.")
            break
        note(f"waiting... (log age {age_h*60:.0f} min)")
        time.sleep(POLL_SECONDS)

    if not CHECKPOINT.exists():
        note("FATAL: no checkpoint at results/checkpoints/pretrain.pt — abort.")
        sys.exit(1)

    with open(ORCH_LOG, "a", encoding="utf-8") as lf:
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_multi_target.py")],
            cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
        ).returncode
    note(f"multi-target orchestrator finished with exit code {rc}")


if __name__ == "__main__":
    main()
