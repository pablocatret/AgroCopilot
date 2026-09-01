import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

from evaluation.reporting import BatchLock


def _workspace_test_root() -> Path:
    root = Path("evaluation/results") / f"test-batch-lock-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_batch_lock_removes_stale_pid_lock():
    root = _workspace_test_root()
    try:
        lock = root / "batch.lock"
        lock.write_text(json.dumps({"pid": 999999999, "created_at": "old"}), encoding="utf-8")

        with BatchLock(root):
            payload = json.loads(lock.read_text(encoding="utf-8"))
            assert payload["pid"] == os.getpid()

        assert not lock.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_batch_lock_blocks_live_pid_lock():
    root = _workspace_test_root()
    try:
        lock = root / "batch.lock"
        lock.write_text(json.dumps({"pid": os.getpid(), "created_at": "now"}), encoding="utf-8")

        with pytest.raises(RuntimeError, match="already locked"):
            with BatchLock(root):
                pass
    finally:
        shutil.rmtree(root, ignore_errors=True)
