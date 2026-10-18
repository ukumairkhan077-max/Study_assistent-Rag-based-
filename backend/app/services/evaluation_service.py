"""Launches the evaluation script as a separate process (it loads models and can
take minutes) and serves its saved JSON results. Nothing here fabricates metrics:
the dashboard only shows files that evaluation/compare_retrieval.py wrote."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from ..core.config import BACKEND_DIR

EVAL_DIR = BACKEND_DIR.parent / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"


class EvaluationService:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._log_tail = ""

    def status(self) -> dict:
        with self._lock:
            if self._proc is None:
                return {"state": "idle", "log_tail": self._log_tail}
            code = self._proc.poll()
            if code is None:
                return {"state": "running", "log_tail": self._log_tail}
            self._log_tail = self._read_log()
            self._proc = None
            return {"state": "finished" if code == 0 else "failed", "exit_code": code, "log_tail": self._log_tail}

    def _read_log(self) -> str:
        f = RESULTS_DIR / "last_run.log"
        return f.read_text(encoding="utf-8", errors="replace")[-2000:] if f.exists() else ""

    def start(self, with_generation: bool = False) -> bool:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return False
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            log = open(RESULTS_DIR / "last_run.log", "w", encoding="utf-8")
            cmd = [sys.executable, str(EVAL_DIR / "compare_retrieval.py")]
            if with_generation:
                cmd.append("--generate")
            self._proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(EVAL_DIR))
            return True

    def latest_results(self) -> Optional[dict]:
        f = RESULTS_DIR / "comparison.json"
        if not f.exists():
            return None
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
