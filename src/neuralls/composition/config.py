"""Composition-layer facade for configuration and profile management."""

from __future__ import annotations

from neuralls.platform.config.loaders import load_raw_toml as load_raw_toml
from neuralls.platform.config.profile import (
    PROFILE_ENV_VAR as PROFILE_ENV_VAR,
    USER_CONFIG_FILE as USER_CONFIG_FILE,
    ProfileConfig as ProfileConfig,
    delete_profile as delete_profile,
    list_profiles as list_profiles,
    load_profile as load_profile,
    resolve_typed_profile as resolve_typed_profile,
    save_profile as save_profile,
)
from neuralls.platform.config.resolution import (
    resolve_case_config_path as resolve_case_config_path,
)
from neuralls.platform.config.settings import (
    CASE_CONFIG_ENV_VAR as CASE_CONFIG_ENV_VAR,
    NeurallsSettings as NeurallsSettings,
    load_case_settings as load_case_settings,
)
