"""Compatibility helpers for importing TensorFlow Hub on Windows."""

from __future__ import annotations

import importlib
import os
import ssl
from types import ModuleType
from typing import Any


def import_tensorflow_hub() -> ModuleType:
    """Import TensorFlow Hub with a verified CA bundle on Windows.

    Some Windows certificate stores contain malformed entries that make
    ``ssl.create_default_context()`` fail before TensorFlow Hub can initialize.
    Supplying certifi explicitly avoids parsing that store without disabling
    TLS certificate verification.
    """
    if os.name != "nt":
        return importlib.import_module("tensorflow_hub")

    try:
        import certifi
    except ImportError as error:
        raise RuntimeError(
            "certifi is required to initialize TensorFlow Hub on Windows"
        ) from error

    original_create_default_context = ssl.create_default_context

    def create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: Any = None,
    ) -> ssl.SSLContext:
        if cafile is None and capath is None and cadata is None:
            cafile = certifi.where()
        return original_create_default_context(
            purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = create_default_context
    try:
        return importlib.import_module("tensorflow_hub")
    finally:
        ssl.create_default_context = original_create_default_context
