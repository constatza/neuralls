"""The runtime environment is collected from the process and flattened to stable `env.*` params."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import pytest
import torch

from neuralls.platform.tracking import runtime_environment
from neuralls.platform.tracking.runtime_environment import (
    RuntimeEnvironment,
    Unreported,
    collect_runtime_environment,
    environment_params,
)

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: Fixture CPU 9000 @ 3.2GHz
processor\t: 1
model name\t: Fixture CPU 9000 @ 3.2GHz
"""


@pytest.fixture
def cpuinfo_file(tmp_path: Path) -> Path:
    path = tmp_path / "cpuinfo"
    path.write_text(CPUINFO, encoding="utf-8")
    return path


@pytest.fixture
def absent_cpuinfo(tmp_path: Path) -> Path:
    return tmp_path / "no-cpuinfo"


@pytest.fixture
def pinned_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix every process-level fact the collector reads except the CPU model."""
    monkeypatch.setattr(runtime_environment.os, "cpu_count", lambda: 12)
    monkeypatch.setattr(runtime_environment.torch, "get_num_threads", lambda: 6)
    monkeypatch.setattr(runtime_environment.torch, "__version__", "2.99.0")
    monkeypatch.setattr(runtime_environment.platform, "python_version", lambda: "3.14.1")
    monkeypatch.setattr(runtime_environment.platform, "processor", lambda: "")
    monkeypatch.setattr(runtime_environment.metadata, "version", lambda name: "0.7.0")


def test_params_of_a_known_environment() -> None:
    env = RuntimeEnvironment(
        device_name="cpu",
        cpu_model="Fixture CPU 9000",
        cpu_logical_count=12,
        torch_threads=6,
        torch_version="2.99.0",
        torchalg_version="0.7.0",
        python_version="3.14.1",
    )

    assert environment_params(env) == {
        "env.device": "cpu",
        "env.cpu_model": "Fixture CPU 9000",
        "env.cpu_logical_count": "12",
        "env.torch_threads": "6",
        "env.torch_version": "2.99.0",
        "env.torchalg_version": "0.7.0",
        "env.python_version": "3.14.1",
    }


def test_unreported_facts_are_recorded_as_unknown() -> None:
    env = RuntimeEnvironment(
        device_name="cpu",
        cpu_model=Unreported.UNKNOWN,
        cpu_logical_count=Unreported.UNKNOWN,
        torch_threads=1,
        torch_version="2.99.0",
        torchalg_version=Unreported.UNKNOWN,
        python_version="3.14.1",
    )

    params = environment_params(env)

    assert (
        params["env.cpu_model"],
        params["env.cpu_logical_count"],
        params["env.torchalg_version"],
    ) == ("unknown", "unknown", "unknown")


@pytest.mark.usefixtures("pinned_process")
def test_collects_the_cpu_environment(cpuinfo_file: Path) -> None:
    env = collect_runtime_environment(torch.device("cpu"), cpuinfo_path=cpuinfo_file)

    assert env == RuntimeEnvironment(
        device_name="cpu",
        cpu_model="Fixture CPU 9000 @ 3.2GHz",
        cpu_logical_count=12,
        torch_threads=6,
        torch_version="2.99.0",
        torchalg_version="0.7.0",
        python_version="3.14.1",
    )


@pytest.mark.usefixtures("pinned_process")
def test_collects_the_cuda_device_name(cpuinfo_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_environment.torch.cuda, "get_device_name", lambda device: "Fixture GPU"
    )

    env = collect_runtime_environment(torch.device("cuda"), cpuinfo_path=cpuinfo_file)

    assert env.device_name == "Fixture GPU"


@pytest.mark.usefixtures("pinned_process")
def test_cpu_model_falls_back_to_the_platform_processor(
    absent_cpuinfo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_environment.platform, "processor", lambda: "Fallback CPU")

    env = collect_runtime_environment(torch.device("cpu"), cpuinfo_path=absent_cpuinfo)

    assert env.cpu_model == "Fallback CPU"


@pytest.mark.usefixtures("pinned_process")
def test_collection_survives_every_unreported_fact(
    absent_cpuinfo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(runtime_environment.os, "cpu_count", lambda: None)
    monkeypatch.setattr(runtime_environment.metadata, "version", missing)

    env = collect_runtime_environment(torch.device("cpu"), cpuinfo_path=absent_cpuinfo)

    assert (env.cpu_model, env.cpu_logical_count, env.torchalg_version) == (
        Unreported.UNKNOWN,
        Unreported.UNKNOWN,
        Unreported.UNKNOWN,
    )
