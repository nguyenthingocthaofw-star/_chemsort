"""Make the local src-layout package importable from repository CLIs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


def ensure_local_package() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


def prepare_tensorflow_environment(model_key: str | None = None) -> None:
    """Set environment variables that must exist before TensorFlow loads.

    EfficientNet-Lite0 is a legacy TF Hub module and needs Keras 2
    (``tf-keras``) under TensorFlow 2.16+.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if model_key == "efficientnet_lite0":
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def configure_tensorflow_logging(tf: Any) -> None:
    """Hide TensorFlow informational/deprecation noise while preserving errors."""
    tf.get_logger().setLevel("ERROR")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
    except ImportError:
        pass


def configure_utf8_console() -> None:
    """Keep Vietnamese output printable on legacy Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
