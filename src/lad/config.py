"""Loads config/sources.yaml and resolves env-based auth references."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "sources.yaml"

load_dotenv()


def load_sources() -> dict[str, dict[str, Any]]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        sources: dict[str, dict[str, Any]] = yaml.safe_load(f)

    for cfg in sources.values():
        auth = cfg.get("auth")
        if isinstance(auth, str) and auth.startswith("env:"):
            env_var = auth.removeprefix("env:")
            value = os.environ.get(env_var)
            cfg["auth_value"] = value
            if not value:
                # No key on hand -- keep the entry visible in `lad status`
                # but don't let a harvest run silently hit an unauthenticated
                # endpoint.
                cfg["enabled"] = False
    return sources


def get_source_config(name: str) -> dict[str, Any]:
    sources = load_sources()
    if name not in sources:
        raise KeyError(f"Unknown source '{name}'. Known sources: {sorted(sources)}")
    return sources[name]
