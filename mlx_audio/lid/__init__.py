import importlib

from .registry import MODEL_REMAPPING

__all__ = ["load", "load_model", "MODEL_REMAPPING"]


def __getattr__(name):
    if name == "utils":
        return importlib.import_module(f"{__name__}.utils")

    if name in {"load", "load_model"}:
        from .utils import load, load_model

        return {"load": load, "load_model": load_model}[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"utils"} | set(__all__))
