"""A fixed session environment for tests that build figure cards, uploaders and report sinks."""

from __future__ import annotations

from neuralls.shared.runtime_environment import RuntimeEnvironment

SESSION_ENVIRONMENT = RuntimeEnvironment(
    device_name="cpu",
    cpu_model="Fixture CPU 9000",
    cpu_logical_count=12,
    torch_threads=6,
    torch_version="2.99.0",
    torchalg_version="0.7.0",
    python_version="3.14.1",
    revision="a" * 40,
    torchalg_revision="b" * 40,
)
"""Every fact reported, so a card round trip shows each one."""
