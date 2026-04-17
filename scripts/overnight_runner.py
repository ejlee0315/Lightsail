"""Overnight orchestration script.

Runs a sequence of experiments with:
- Checkpointing (know what's done)
- Auto-restart on crash (up to 2 retries per experiment)
- Timeout per experiment (90 min max)
- Progress logging to `overnight_status.json`

Usage:
    python3 scripts/overnight_runner.py

This runs in the foreground but streams output to a log file so you can
tail it externally. Designed to be started via nohup:

    nohup python3 scripts/overnight_runner.py > overnight.log 2>&1 &
"""
from __future__ import annotations
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATUS_FILE = ROOT / "overnight_status.json"
LOG_FILE = ROOT / "overnight.log"

# Maximum runtime per experiment in seconds (90 min)
MAX_TIMEOUT_SEC = 90 * 60
# Number of retry attempts per experiment
MAX_RETRIES = 2

EXPERIMENTS = [
    # (id, cmd, est_minutes, priority)
    {
        "id": "1320_thin_s42",
        "cmd": ["python3", "scripts/run_stage1_1320_thin.py", "--seed", "42"],
        "est_min": 50,
    },
    {
        "id": "1550_mfs500_s42",
        "cmd": ["python3", "scripts/run_mfs500_sweep.py", "--seed", "42", "--launch", "1550"],
        "est_min": 50,
    },
    {
        "id": "1550_mfs500_s123",
        "cmd": ["python3", "scripts/run_mfs500_sweep.py", "--seed", "123", "--launch", "1550"],
        "est_min": 50,
    },
    {
        "id": "1550_mfs500_s456",
        "cmd": ["python3", "scripts/run_mfs500_sweep.py", "--seed", "456", "--launch", "1550"],
        "est_min": 50,
    },
    {
        "id": "1320_mfs500_s42",
        "cmd": ["python3", "scripts/run_mfs500_sweep.py", "--seed", "42", "--launch", "1320"],
        "est_min": 50,
    },
    {
        "id": "fab_tolerance",
        "cmd": ["python3", "scripts/run_fab_tolerance.py"],
        "est_min": 60,
    },
    {
        "id": "1550_thin_s123",
        "cmd": ["python3", "scripts/run_stage1_1550_1850.py", "--seed", "123"],
        "est_min": 50,
    },
]


def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {"started": datetime.now().isoformat(), "experiments": {}}


def save_status(status: dict) -> None:
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def run_experiment(exp: dict, status: dict) -> bool:
    eid = exp["id"]
    attempts = status["experiments"].get(eid, {}).get("attempts", 0)

    if attempts >= MAX_RETRIES + 1:
        logging.warning("Exp %s: exceeded retries, skipping", eid)
        return False

    logging.info("Starting %s (attempt %d, est %d min)...",
                 eid, attempts + 1, exp["est_min"])

    status["experiments"][eid] = {
        "state": "running",
        "started": datetime.now().isoformat(),
        "attempts": attempts + 1,
        "est_min": exp["est_min"],
    }
    save_status(status)

    t0 = time.time()
    try:
        result = subprocess.run(
            exp["cmd"],
            cwd=ROOT,
            timeout=MAX_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            logging.info("  %s DONE in %.1f min", eid, elapsed / 60)
            status["experiments"][eid].update({
                "state": "done",
                "finished": datetime.now().isoformat(),
                "elapsed_min": elapsed / 60,
                "returncode": 0,
            })
            # Extract last output line from stdout
            last_lines = [l for l in result.stdout.strip().split("\n")[-5:] if l]
            status["experiments"][eid]["last_output"] = "\n".join(last_lines)
            save_status(status)
            return True
        else:
            logging.error("  %s FAILED rc=%d in %.1f min",
                          eid, result.returncode, elapsed / 60)
            status["experiments"][eid].update({
                "state": "failed",
                "finished": datetime.now().isoformat(),
                "elapsed_min": elapsed / 60,
                "returncode": result.returncode,
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            })
            save_status(status)
            return False

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        logging.error("  %s TIMED OUT after %.1f min", eid, elapsed / 60)
        status["experiments"][eid].update({
            "state": "timeout",
            "finished": datetime.now().isoformat(),
            "elapsed_min": elapsed / 60,
        })
        save_status(status)
        return False
    except Exception as exc:
        logging.exception("  %s CRASHED: %s", eid, exc)
        status["experiments"][eid].update({
            "state": "crashed",
            "finished": datetime.now().isoformat(),
            "error": str(exc),
        })
        save_status(status)
        return False


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("=" * 72)
    logging.info("OVERNIGHT RUNNER started at %s", datetime.now().isoformat())
    logging.info("Total experiments: %d", len(EXPERIMENTS))
    logging.info("Est total: %d min", sum(e["est_min"] for e in EXPERIMENTS))
    logging.info("=" * 72)

    status = load_status()

    # Run until all done or retries exhausted
    for round_num in range(MAX_RETRIES + 1):
        todo = []
        for exp in EXPERIMENTS:
            eid = exp["id"]
            st = status["experiments"].get(eid, {})
            if st.get("state") != "done":
                todo.append(exp)

        if not todo:
            logging.info("All experiments done!")
            break

        logging.info("Round %d: %d experiments to run", round_num + 1, len(todo))

        for exp in todo:
            run_experiment(exp, status)

    # Final summary
    done = sum(1 for e in EXPERIMENTS
               if status["experiments"].get(e["id"], {}).get("state") == "done")
    logging.info("=" * 72)
    logging.info("FINAL: %d/%d done", done, len(EXPERIMENTS))
    logging.info("=" * 72)
    status["finished"] = datetime.now().isoformat()
    save_status(status)


if __name__ == "__main__":
    main()
