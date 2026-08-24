"""Immutable input snapshots with end-of-run source stability checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import tempfile
from types import TracebackType


_COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    size_bytes: int
    modification_time_ns: int
    change_time_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_FileIdentity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=stat.S_IFMT(value.st_mode),
            size_bytes=value.st_size,
            modification_time_ns=value.st_mtime_ns,
            # Windows may report a slightly different creation/change timestamp
            # for fstat(handle) and stat(path) on the same unchanged file. The
            # stable volume/file identifiers plus size, mtime and content hash
            # still detect replacement and mutation without that false positive.
            change_time_ns=0 if os.name == "nt" else value.st_ctime_ns,
        )


@dataclass(frozen=True)
class InputSnapshot:
    """One immutable copy and the identity of the source that produced it."""

    source_path: Path
    snapshot_path: Path
    size_bytes: int
    sha256: str
    _source_identity: _FileIdentity

    def report_record(self) -> dict[str, object]:
        return {
            "path": str(self.source_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


class InputSnapshotSet:
    """Own a private set of per-file snapshots for one evaluation."""

    def __init__(self) -> None:
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._snapshots: dict[Path, InputSnapshot] = {}

    def __enter__(self) -> "InputSnapshotSet":
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="plascan_eth3d_depth_eval_"
        )
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception_type, exception, traceback
        if self._temporary_directory is not None:
            for snapshot in self._snapshots.values():
                try:
                    snapshot.snapshot_path.chmod(
                        stat.S_IRUSR | stat.S_IWUSR
                    )
                except OSError:
                    pass
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._snapshots.clear()

    def capture(self, path: Path) -> InputSnapshot:
        """Copy one stable source read into a suffix-preserving private file."""

        source_path = path.resolve()
        existing = self._snapshots.get(source_path)
        if existing is not None:
            return existing
        if self._temporary_directory is None:
            raise RuntimeError("InputSnapshotSet must be entered before capture")

        snapshot_root = Path(self._temporary_directory.name)
        snapshot_path = snapshot_root / (
            f"{len(self._snapshots):03d}_{source_path.name}"
        )
        digest = hashlib.sha256()
        copied_bytes = 0
        try:
            with source_path.open("rb") as source, snapshot_path.open("xb") as target:
                identity_before = _FileIdentity.from_stat(os.fstat(source.fileno()))
                if identity_before.mode != stat.S_IFREG:
                    raise ValueError(f"Input is not a regular file: {source_path}")
                while chunk := source.read(_COPY_CHUNK_BYTES):
                    target.write(chunk)
                    digest.update(chunk)
                    copied_bytes += len(chunk)
                identity_after = _FileIdentity.from_stat(os.fstat(source.fileno()))
                target.flush()
                os.fsync(target.fileno())
        except FileNotFoundError as error:
            snapshot_path.unlink(missing_ok=True)
            raise FileNotFoundError(
                f"Input file not found: {source_path}"
            ) from error
        except OSError as error:
            snapshot_path.unlink(missing_ok=True)
            raise OSError(f"Could not snapshot input {source_path}: {error}") from error

        if identity_before != identity_after:
            snapshot_path.unlink(missing_ok=True)
            raise ValueError(f"Input changed while taking snapshot: {source_path}")
        if copied_bytes != identity_before.size_bytes:
            snapshot_path.unlink(missing_ok=True)
            raise ValueError(
                f"Input size changed while taking snapshot: {source_path}"
            )
        path_identity = _identity_from_path(source_path)
        if path_identity != identity_before:
            snapshot_path.unlink(missing_ok=True)
            raise ValueError(f"Input path changed while taking snapshot: {source_path}")

        snapshot_path.chmod(stat.S_IRUSR)
        snapshot = InputSnapshot(
            source_path=source_path,
            snapshot_path=snapshot_path,
            size_bytes=copied_bytes,
            sha256=digest.hexdigest(),
            _source_identity=identity_before,
        )
        self._snapshots[source_path] = snapshot
        return snapshot

    def verify_unchanged(self) -> None:
        """Reject content, metadata, inode, and detectable ABA source changes."""

        for snapshot in self._snapshots.values():
            _verify_source_unchanged(snapshot)


def _identity_from_path(path: Path) -> _FileIdentity:
    try:
        return _FileIdentity.from_stat(path.stat())
    except OSError as error:
        raise ValueError(f"Input is unavailable after snapshot: {path}") from error


def _verify_source_unchanged(snapshot: InputSnapshot) -> None:
    source_path = snapshot.source_path
    if _identity_from_path(source_path) != snapshot._source_identity:
        raise ValueError(f"Input changed during evaluation: {source_path}")

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source_path.open("rb") as stream:
            identity_before = _FileIdentity.from_stat(os.fstat(stream.fileno()))
            if identity_before != snapshot._source_identity:
                raise ValueError(
                    f"Input changed during evaluation: {source_path}"
                )
            while chunk := stream.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
                byte_count += len(chunk)
            identity_after = _FileIdentity.from_stat(os.fstat(stream.fileno()))
    except OSError as error:
        raise ValueError(
            f"Input is unavailable after snapshot: {source_path}"
        ) from error

    if (
        identity_after != snapshot._source_identity
        or byte_count != snapshot.size_bytes
        or digest.hexdigest() != snapshot.sha256
    ):
        raise ValueError(f"Input changed during evaluation: {source_path}")
