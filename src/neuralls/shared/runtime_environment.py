"""The execution environment every timing of a comparison session depends on.

A wall time is only comparable with another taken on the same device, CPU, thread count,
library versions and code revision, so those facts are constant for one session. The session
run records them once as parameters; reports read them back and cite them in each figure card.
The record and its string codec live here because both the MLflow layer and the figure cards
use it, and neither may import the other.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class Unreported(StrEnum):
    """A fact the platform could not report, recorded as an explicit value rather than omitted."""

    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Execution environment of one comparison session.

    Every fact may be `Unreported`: collection reports what it could read, and a session
    stored before a fact was recorded has no value for it.

    Attributes:
        device_name: CUDA device name, or `CPU_DEVICE_NAME` when the solves run on the CPU.
        cpu_model: Host CPU model string.
        cpu_logical_count: Logical CPUs of the host.
        torch_threads: Intra-op thread count torch uses for CPU kernels.
        torch_version: Installed torch version.
        torchalg_version: Installed `torchalg` version.
        python_version: Interpreter version.
        revision: Git revision of the neuralls checkout, suffixed `-dirty` when it had
            uncommitted changes.
        torchalg_revision: Git commit the installed `torchalg` was built from.
    """

    device_name: str | Unreported
    cpu_model: str | Unreported
    cpu_logical_count: int | Unreported
    torch_threads: int | Unreported
    torch_version: str | Unreported
    torchalg_version: str | Unreported
    python_version: str | Unreported
    revision: str | Unreported
    torchalg_revision: str | Unreported


def encode_environment(env: RuntimeEnvironment) -> dict[str, str]:
    """The environment as stable string fields, one per fact."""
    return {
        "device": str(env.device_name),
        "cpu_model": str(env.cpu_model),
        "cpu_logical_count": str(env.cpu_logical_count),
        "torch_threads": str(env.torch_threads),
        "torch_version": str(env.torch_version),
        "torchalg_version": str(env.torchalg_version),
        "python_version": str(env.python_version),
        "revision": str(env.revision),
        "torchalg_revision": str(env.torchalg_revision),
    }


def _text(fields: Mapping[str, str], key: str) -> str | Unreported:
    text = fields.get(key, Unreported.UNKNOWN)
    return Unreported.UNKNOWN if text == Unreported.UNKNOWN else text


def _count(fields: Mapping[str, str], key: str) -> int | Unreported:
    text = fields.get(key, Unreported.UNKNOWN)
    return int(text) if text.isdecimal() else Unreported.UNKNOWN


def decode_environment(fields: Mapping[str, str]) -> RuntimeEnvironment:
    """Rebuild an environment from `encode_environment` fields; a missing fact is unreported."""
    return RuntimeEnvironment(
        device_name=_text(fields, "device"),
        cpu_model=_text(fields, "cpu_model"),
        cpu_logical_count=_count(fields, "cpu_logical_count"),
        torch_threads=_count(fields, "torch_threads"),
        torch_version=_text(fields, "torch_version"),
        torchalg_version=_text(fields, "torchalg_version"),
        python_version=_text(fields, "python_version"),
        revision=_text(fields, "revision"),
        torchalg_revision=_text(fields, "torchalg_revision"),
    )
