"""Verified acquisition of external ChessFly artifacts from Hugging Face."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.request import Request, urlopen

from platformdirs import user_cache_path

SPACE_BASE = "https://huggingface.co/spaces/mlabonne/chessfly/resolve"
MODEL_BASE = "https://huggingface.co/mlabonne/chessfly/resolve"


class ArtifactError(RuntimeError):
    """Raised when an upstream artifact cannot be acquired or verified."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.name.endswith(".gz"):
        try:
            return gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            raise ArtifactError(f"could not decompress {path.name}") from exc
    return raw


def default_cache_dir() -> Path:
    return Path(user_cache_path("chessfly-statepings")) / "artifacts"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    root: Path
    weights: Path
    connectome: Path
    neurons: Path
    manifest: Path

    @classmethod
    def under(cls, root: str | Path) -> "ArtifactPaths":
        root = Path(root)
        return cls(
            root=root,
            weights=root / "flynet.safetensors",
            connectome=root / "connectome.bin.gz",
            neurons=root / "neurons.bin.gz",
            manifest=root / "manifest.json",
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    name: str
    source_url: str
    requested_revision: str
    resolved_revision: str | None
    bytes: int
    sha256: str
    decoded_sha256: str | None = None
    expected_decoded_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    acquired_at_utc: str
    dataset: str
    space_revision: str
    model_revision: str
    files: tuple[ArtifactRecord, ...]
    root: str

    @property
    def paths(self) -> ArtifactPaths:
        return ArtifactPaths.under(self.root)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["files"] = [asdict(item) for item in self.files]
        return result

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)

    @classmethod
    def read(cls, path: Path) -> "ArtifactManifest":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                acquired_at_utc=str(payload["acquired_at_utc"]),
                dataset=str(payload["dataset"]),
                space_revision=str(payload["space_revision"]),
                model_revision=str(payload["model_revision"]),
                files=tuple(ArtifactRecord(**item) for item in payload["files"]),
                root=str(payload["root"]),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ArtifactError(f"invalid artifact manifest {path}") from exc

    def verify(self) -> None:
        paths = self.paths
        by_name = {item.name: item for item in self.files}
        for name, path in (
            ("flynet.safetensors", paths.weights),
            ("connectome.bin.gz", paths.connectome),
            ("neurons.bin.gz", paths.neurons),
        ):
            record = by_name.get(name)
            if record is None or not path.is_file():
                raise ArtifactError(f"missing cached artifact {name}")
            if path.stat().st_size != record.bytes or _sha256_path(path) != record.sha256:
                raise ArtifactError(f"cached artifact verification failed for {name}")
            if record.expected_decoded_sha256:
                actual = _sha256_bytes(_decoded(path))
                if actual != record.expected_decoded_sha256:
                    raise ArtifactError(f"decoded SHA-256 mismatch for {name}")


def _fetch_bytes(url: str) -> tuple[bytes, Mapping[str, str]]:
    request = Request(url, headers={"User-Agent": "ChessFlyStatePings/0.1"})
    try:
        with urlopen(request) as response:
            data = response.read()
            headers = {key.lower(): value for key, value in response.headers.items()}
            return data, headers
    except Exception as exc:
        raise ArtifactError(f"download failed for {url}: {exc}") from exc


def _atomic_write_verified(destination: Path, payload: bytes, *, expected_decoded_sha256: str | None = None) -> tuple[str, str | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        decoded_sha = None
        if expected_decoded_sha256 is not None:
            raw = temp.read_bytes()
            decoded = gzip.decompress(raw) if destination.name.endswith(".gz") else raw
            decoded_sha = _sha256_bytes(decoded)
            if decoded_sha != expected_decoded_sha256:
                raise ArtifactError(f"decoded SHA-256 mismatch for {destination.name}")
        digest = _sha256_path(temp)
        os.replace(temp, destination)
        return digest, decoded_sha
    finally:
        temp.unlink(missing_ok=True)


def ensure_artifacts(cache_dir: str | Path | None = None, *, space_revision: str = "main", model_revision: str = "main", force: bool = False) -> ArtifactManifest:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    paths = ArtifactPaths.under(root)
    root.mkdir(parents=True, exist_ok=True)
    if paths.manifest.is_file() and not force:
        manifest = ArtifactManifest.read(paths.manifest)
        if Path(manifest.root).resolve() == root.resolve():
            manifest.verify()
            return manifest
    meta_url = f"{SPACE_BASE}/{space_revision}/data/meta.json"
    meta_bytes, meta_headers = _fetch_bytes(meta_url)
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError("invalid ChessFly Space metadata") from exc
    published = meta.get("files", {})
    specs = [
        ("connectome.bin.gz", paths.connectome, f"{SPACE_BASE}/{space_revision}/data/connectome.bin.gz", space_revision, published.get("connectome.bin.gz", {}).get("rawSha256")),
        ("neurons.bin.gz", paths.neurons, f"{SPACE_BASE}/{space_revision}/data/neurons.bin.gz", space_revision, published.get("neurons.bin.gz", {}).get("rawSha256")),
        ("flynet.safetensors", paths.weights, f"{MODEL_BASE}/{model_revision}/flynet.safetensors?download=true", model_revision, None),
    ]
    records: list[ArtifactRecord] = []
    try:
        for name, destination, url, requested, expected_raw in specs:
            payload, headers = _fetch_bytes(url)
            digest, decoded_sha = _atomic_write_verified(destination, payload, expected_decoded_sha256=expected_raw)
            records.append(ArtifactRecord(name=name, source_url=url, requested_revision=requested, resolved_revision=headers.get("x-repo-commit") or meta_headers.get("x-repo-commit"), bytes=len(payload), sha256=digest, decoded_sha256=decoded_sha, expected_decoded_sha256=expected_raw))
    except Exception:
        paths.manifest.unlink(missing_ok=True)
        raise
    manifest = ArtifactManifest(acquired_at_utc=datetime.now(timezone.utc).isoformat(), dataset=str(meta.get("dataset", "unknown")), space_revision=space_revision, model_revision=model_revision, files=tuple(records), root=str(root.resolve()))
    manifest.verify()
    manifest.write(paths.manifest)
    return manifest


__all__ = ["ArtifactError", "ArtifactManifest", "ArtifactPaths", "ArtifactRecord", "default_cache_dir", "ensure_artifacts"]
