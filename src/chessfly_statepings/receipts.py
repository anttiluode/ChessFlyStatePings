"""Small JSON experiment receipts."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
    except Exception:
        return None


def build_receipt(*, command: str, arguments: Mapping[str, Any], device: str, artifact_manifest: Any, model_metadata: Mapping[str, Any], inputs: Sequence[Any], results: Any, instability_count: int) -> dict[str, Any]:
    artifacts = artifact_manifest.to_dict() if hasattr(artifact_manifest, "to_dict") else artifact_manifest
    return {
        "schema": "chessfly-statepings-receipt-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "command": command,
        "arguments": dict(arguments),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "python_chess": _version("python-chess"),
            "safetensors": _version("safetensors"),
        },
        "device": device,
        "artifacts": artifacts,
        "model": dict(model_metadata),
        "inputs": list(inputs),
        "results": results,
        "instability_count": int(instability_count),
    }


def write_receipt(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


__all__ = ["build_receipt", "write_receipt"]
