"""Guard tests: importing any module must not download data or load TF.

These protect against a regression to the old behaviour where importing the
entry-point scripts kicked off a yfinance download, a TensorFlow model load,
an APScheduler job, and a busy loop.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest

# The entry-point scripts (main.py, app.py) live at the repository root rather
# than inside the package, so make sure that directory is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PURE_MODULES = [
    "stockforecast",
    "stockforecast.features",
    "stockforecast.data_shaping",
]

# These import package-level symbols but must still avoid heavy/IO imports
# until a function is actually called.
ENTRY_MODULES = [
    "stockforecast.data",
    "stockforecast.model",
    "main",
    "app",
]

FORBIDDEN_ON_IMPORT = {"yfinance", "tensorflow", "streamlit", "apscheduler"}


@pytest.fixture
def block_heavy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make importing any forbidden heavy module raise immediately."""
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        root = name.split(".")[0]
        if root in FORBIDDEN_ON_IMPORT:
            raise AssertionError(f"module import triggered heavy import: {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("module_name", PURE_MODULES + ENTRY_MODULES)
def test_import_does_not_trigger_heavy_imports(module_name: str, block_heavy_imports: None) -> None:
    for heavy in FORBIDDEN_ON_IMPORT:
        sys.modules.pop(heavy, None)
    sys.modules.pop(module_name, None)
    # Should import cleanly without pulling in yfinance/tensorflow/etc.
    importlib.import_module(module_name)


@pytest.mark.parametrize("module_name", PURE_MODULES + ENTRY_MODULES)
def test_modules_import_at_all(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    assert module is not None
