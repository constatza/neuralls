"""Collect and log the execution environment every timing of a comparison session depends on.

The facts are constant for one session, so they are recorded once as parameters of the session
run, never per row or in a parquet table (see `shared.runtime_environment`).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import torch

from neuralls.shared.runtime_environment import (
    RuntimeEnvironment,
    Unreported,
    decode_environment,
    encode_environment,
)

TORCHALG_DISTRIBUTION = "torchalg"
"""Name of the installed distribution whose version is recorded."""

CPUINFO_PATH = Path("/proc/cpuinfo")
"""Linux source of the CPU model string; `platform.processor()` only names the architecture there."""

CPUINFO_MODEL_KEY = "model name"
"""Key of the CPU model line in `/proc/cpuinfo`."""

CPU_DEVICE_NAME = "cpu"
"""Device name recorded when the solves run on the CPU."""

ENV_PARAM_PREFIX = "env."
"""Prefix of the session run params that hold the environment facts."""

PACKAGE_DIRECTORY = Path(__file__).resolve().parent
"""A directory inside the neuralls checkout, where its git revision is read."""

GIT_TIMEOUT_SECONDS = 10
"""Upper bound on the git call that reads the checkout revision."""

DIRECT_URL_FILE = "direct_url.json"
"""Distribution metadata file recording the VCS commit a package was installed from."""


def environment_params(env: RuntimeEnvironment) -> dict[str, str]:
    """The MLflow params of a runtime environment; the `env.` keys are stable identifiers."""
    return {f"{ENV_PARAM_PREFIX}{key}": value for key, value in encode_environment(env).items()}


def read_environment_params(params: Mapping[str, str]) -> RuntimeEnvironment:
    """The environment a session run logged; a session that did not log a fact reports it unknown."""
    fields = {
        key.removeprefix(ENV_PARAM_PREFIX): value
        for key, value in params.items()
        if key.startswith(ENV_PARAM_PREFIX)
    }
    return decode_environment(fields)


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


def _git_revision(directory: Path) -> str | Unreported:
    """`git describe --always --dirty` of the checkout containing `directory`, if there is one."""
    try:
        done = subprocess.run(
            ["git", "-C", str(directory), "describe", "--always", "--dirty", "--abbrev=40"],
            capture_output=True,
            encoding="utf-8",
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except OSError, subprocess.TimeoutExpired:
        return Unreported.UNKNOWN
    if done.returncode != 0:
        return Unreported.UNKNOWN
    return done.stdout.strip() or Unreported.UNKNOWN


def _torchalg_revision() -> str | Unreported:
    """The git commit `torchalg` was installed from, or `Unreported` when it was not a VCS install."""
    try:
        files = metadata.distribution(TORCHALG_DISTRIBUTION).files or ()
        record = next((file for file in files if file.name == DIRECT_URL_FILE), None)
        if record is None:
            return Unreported.UNKNOWN
        commit = json.loads(record.read_text(encoding="utf-8"))["vcs_info"]["commit_id"]
    except metadata.PackageNotFoundError, OSError, ValueError, KeyError, TypeError:
        return Unreported.UNKNOWN
    return commit if isinstance(commit, str) and commit else Unreported.UNKNOWN


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
        revision=_git_revision(PACKAGE_DIRECTORY),
        torchalg_revision=_torchalg_revision(),
    )
