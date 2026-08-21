"""Prepare and atomically publish evaluator output artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np


@dataclass
class PreparedOutput:
    final_path: Path
    temporary_path: Path
    size_bytes: int
    sha256: str

    def report_record(self) -> dict[str, object]:
        return {
            "path": str(self.final_path),
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def cleanup(self) -> None:
        self.temporary_path.unlink(missing_ok=True)


def prepare_npy_output(path: Path, values: np.ndarray) -> PreparedOutput:
    _require_absolute_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        return _prepared_output(path, temporary_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def prepare_text_output(path: Path, value: str) -> PreparedOutput:
    _require_absolute_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        return _prepared_output(path, temporary_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def publish_outputs(
    report: PreparedOutput,
    remapped_ground_truth: PreparedOutput | None,
    *,
    overwrite: bool,
) -> None:
    """Publish report last with explicit no-clobber/overwrite recovery."""

    if remapped_ground_truth is None:
        _publish_single(report, overwrite=overwrite)
        return
    if overwrite:
        _publish_pair_with_overwrite(report, remapped_ground_truth)
    else:
        _publish_pair_without_clobber(report, remapped_ground_truth)


def _prepared_output(final_path: Path, temporary_path: Path) -> PreparedOutput:
    digest = hashlib.sha256()
    byte_count = 0
    with temporary_path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return PreparedOutput(
        final_path=final_path,
        temporary_path=temporary_path,
        size_bytes=byte_count,
        sha256=digest.hexdigest(),
    )


def _publish_single(output: PreparedOutput, *, overwrite: bool) -> None:
    if overwrite:
        os.replace(output.temporary_path, output.final_path)
    else:
        _link_no_clobber(output)
    output.cleanup()


def _publish_pair_without_clobber(
    report: PreparedOutput,
    remapped_ground_truth: PreparedOutput,
) -> None:
    remapped_published = False
    try:
        _link_no_clobber(remapped_ground_truth)
        remapped_published = True
        _link_no_clobber(report)
    except BaseException as error:
        if remapped_published:
            raise RuntimeError(
                "Report publication failed after remapped GT was committed; "
                "the no-clobber orphan was retained for audit at "
                f"{remapped_ground_truth.final_path}"
            ) from error
        raise
    else:
        remapped_ground_truth.cleanup()
        report.cleanup()


def _publish_pair_with_overwrite(
    report: PreparedOutput,
    remapped_ground_truth: PreparedOutput,
) -> None:
    backup_path = _backup_existing_file(remapped_ground_truth.final_path)
    remapped_published = False
    try:
        os.replace(
            remapped_ground_truth.temporary_path,
            remapped_ground_truth.final_path,
        )
        remapped_published = True
        os.replace(report.temporary_path, report.final_path)
    except BaseException:
        if remapped_published:
            try:
                _restore_previous_file(remapped_ground_truth.final_path, backup_path)
                backup_path = None
            except BaseException as rollback_error:
                raise RuntimeError(
                    "Report publication failed and remapped-GT rollback also failed"
                ) from rollback_error
        raise
    finally:
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
    report.cleanup()
    remapped_ground_truth.cleanup()


def _link_no_clobber(output: PreparedOutput) -> None:
    try:
        os.link(output.temporary_path, output.final_path)
    except FileExistsError as error:
        raise FileExistsError(
            f"Output already exists; use --overwrite: {output.final_path}"
        ) from error


def _backup_existing_file(path: Path) -> Path | None:
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_symlink() and not path.is_file():
        raise ValueError(f"Output path is not a regular file: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.backup.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    backup_path = Path(name)
    backup_path.unlink()
    try:
        if path.is_symlink():
            os.symlink(
                os.readlink(path),
                backup_path,
                target_is_directory=path.is_dir(),
            )
        else:
            os.link(path, backup_path)
    except BaseException:
        backup_path.unlink(missing_ok=True)
        raise
    return backup_path


def _restore_previous_file(path: Path, backup_path: Path | None) -> None:
    if backup_path is None:
        path.unlink(missing_ok=True)
    else:
        os.replace(backup_path, path)


def _require_absolute_output_path(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError(
            "Prepared output paths must be absolute and frozen before use: "
            f"{path}"
        )
