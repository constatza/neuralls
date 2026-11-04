"""The execution environment every timing of a comparison session depends on.

A wall time is only comparable with another taken on the same device, CPU, thread count and
library versions, so those facts are constant for one session and are recorded once as
parameters of the session run, never per row or in a parquet table.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from enum import StrEnum
from importlib import metadata
from pathlib import Path

import torch

TORCHALG_DISTRIBUTION = "torchalg"
"""Name of the installed distribution whose version is recorded."""

CPUINFO_PATH = Path("/proc/cpuinfo")
"""Linux source of the CPU model string; `platform.processor()` only names the architecture there."""

CPUINFO_MODEL_KEY = "model name"
"""Key of the CPU model line in `/proc/cpuinfo`."""

CPU_DEVICE_NAME = "cpu"
"""Device name recorded when the solves run on the CPU."""


class Unreported(StrEnum):
    """A fact the platform could not report, recorded as an explicit value rather than omitted."""

    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RuntimeEnvironment:
    """Execution environment of one comparison session.

    Attributes:
        device_name: CUDA device name, or `CPU_DEVICE_NAME` when the solves run on the CPU.
        cpu_model: Host CPU model string.
        cpu_logical_count: Logical CPUs of the host.
        torch_threads: Intra-op thread count torch uses for CPU kernels.
        torch_version: Installed torch version.
        torchalg_version: Installed `torchalg` version.
        python_version: Interpreter version.
    """

    device_name: str
    cpu_model: str | Unreported
    cpu_logical_count: int | Unreported
    torch_threads: int
    torch_version: str
    torchalg_version: str | Unreported
    python_version: str


def environment_params(env: RuntimeEnvironment) -> dict[str, str]:
    """The MLflow params of a runtime environment; the `env.` keys are stable identifiers."""
    return {
        "env.device": env.device_name,
        "env.cpu_model": str(env.cpu_model),
        "env.cpu_logical_count": str(env.cpu_logical_count),
        "env.torch_threads": str(env.torch_threads),
        "env.torch_version": env.torch_version,
        "env.torchalg_version": str(env.torchalg_version),
        "env.python_version": env.python_version,
    }


def _cpuinfo_model(cpuinfo_path: Path) -> str | Unreported:
    """The first `model name` line of a cpuinfo file, or `Unreported` when there is none."""
    try:
        lines = cpuinfo_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return Unreported.UNKNOWN
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip() == CPUINFO_MODEL_KEY and value.strip():
            return value.strip()
    return Unreported.UNKNOWN


def _cpu_model(cpuinfo_path: Path) -> str | Unreported:
    """The CPU model: cpuinfo where it exists, else `platform.processor()`, else `Unreported`."""
    from_cpuinfo = _cpuinfo_model(cpuinfo_path)
    if from_cpuinfo is not Unreported.UNKNOWN:
        return from_cpuinfo
    return platform.processor().strip() or Unreported.UNKNOWN


def _torchalg_version() -> str | Unreported:
    """The installed `torchalg` version, or `Unreported` when it is not an installed distribution."""
    try:
        return metadata.version(TORCHALG_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return Unreported.UNKNOWN


def _device_name(device: torch.device) -> str:
    """The CUDA device name, or `CPU_DEVICE_NAME` for any non-CUDA device."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return CPU_DEVICE_NAME


def collect_runtime_environment(
    device: torch.device, *, cpuinfo_path: Path = CPUINFO_PATH
) -> RuntimeEnvironment:
    """Read the environment of the running process; never raises for an unreported fact.

    Args:
        device: Device the session's solves run on.
        cpuinfo_path: Linux cpuinfo file to read the CPU model from.
    """
    return RuntimeEnvironment(
        device_name=_device_name(device),
        cpu_model=_cpu_model(cpuinfo_path),
        cpu_logical_count=os.cpu_count() or Unreported.UNKNOWN,
        torch_threads=torch.get_num_threads(),
        torch_version=torch.__version__,
        torchalg_version=_torchalg_version(),
        python_version=platform.python_version(),
    )
