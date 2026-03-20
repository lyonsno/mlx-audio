"""Lightweight model-routing helpers shared across MLX Audio subsystems."""

import importlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from mlx_audio.lid.registry import MODEL_REMAPPING as LID_MODEL_REMAPPING
from mlx_audio.stt.registry import MODEL_REMAPPING as STT_MODEL_REMAPPING
from mlx_audio.tts.registry import MODEL_REMAPPING as TTS_MODEL_REMAPPING
from mlx_audio.vad.registry import MODEL_REMAPPING as VAD_MODEL_REMAPPING


def is_valid_module_name(name: str) -> bool:
    """Check if a string is a valid Python module name."""
    if not name or not isinstance(name, str):
        return False

    return name[0].isalpha() or name[0] == "_"


def _allows_direct_category_match(
    category: str, hint: str, candidates: List[str]
) -> bool:
    """Gate ambiguous direct matches that need an explicit repo-name hint."""
    if category == "lid" and hint == "wav2vec2" and "lid" not in candidates:
        return False

    return True


def _get_available_models(category: str) -> set[str]:
    models_dir = Path(__file__).parent / category / "models"
    if not models_dir.exists():
        return set()

    return {
        item.name
        for item in models_dir.iterdir()
        if item.is_dir() and not item.name.startswith("__")
    }


def get_model_category(model_type: str, model_name: List[str]) -> Optional[str]:
    """Determine a model category without importing model implementations."""
    candidates = [model_type] + (model_name or [])
    categories = [
        ("tts", TTS_MODEL_REMAPPING),
        ("stt", STT_MODEL_REMAPPING),
        ("lid", LID_MODEL_REMAPPING),
        ("vad", VAD_MODEL_REMAPPING),
    ]
    available_models = {
        category: _get_available_models(category) for category, _ in categories
    }

    for category, remap in categories:
        if model_type not in remap:
            continue

        arch = remap[model_type]
        if not is_valid_module_name(arch) or arch not in available_models[category]:
            continue

        return category

    if is_valid_module_name(model_type):
        direct_model_type_matches = [
            category
            for category, _ in categories
            if model_type in available_models[category]
            and _allows_direct_category_match(category, model_type, candidates)
        ]
        if len(direct_model_type_matches) == 1:
            return direct_model_type_matches[0]

    for category, remap in categories:
        for hint in model_name or []:
            if hint in remap:
                arch = remap[hint]
                if (
                    not is_valid_module_name(arch)
                    or arch not in available_models[category]
                ):
                    continue
                return category

    for category, remap in categories:
        for hint in model_name or []:
            if (
                hint not in remap
                and is_valid_module_name(hint)
                and hint in available_models[category]
                and _allows_direct_category_match(category, hint, candidates)
            ):
                return category

    return None


def get_model_class(
    model_type: str,
    model_name: List[str],
    category: str,
    model_remapping: dict,
) -> Tuple:
    """Retrieve the architecture module for a resolved category/model family."""
    model_type_mapped = model_remapping.get(model_type, None)
    available_models = _get_available_models(category)
    resolved_model_type = model_type

    if model_name is not None and model_type_mapped != model_type:
        matched_from_name = False
        for part in model_name:
            if part in available_models:
                resolved_model_type = part
                matched_from_name = True
            if part in model_remapping:
                resolved_model_type = model_remapping[part]
                matched_from_name = True
                break
        if not matched_from_name and model_type_mapped is not None:
            resolved_model_type = model_type_mapped
    elif model_type_mapped is not None:
        resolved_model_type = model_type_mapped

    model_type = resolved_model_type

    try:
        module_path = f"mlx_audio.{category}.models.{model_type}"
        arch = importlib.import_module(module_path)
    except ImportError as e:
        if e.name != module_path:
            print("\n", flush=True)

            raise ImportError(
                f"\nMissing dependency while loading {model_type}: {e}\n"
                f"Please install it using: pip install {e.name}"
            ) from e

        msg = f"Model type {model_type} not supported for {category}."
        logging.error(msg)
        raise ValueError(msg)

    return arch, model_type


def get_model_name_parts(model_path) -> List[str]:
    if isinstance(model_path, str):
        return model_path.lower().split("/")[-1].split("-")

    if isinstance(model_path, Path):
        index = model_path.parts.index("hub")
        return model_path.parts[index + 1].lower().split("--")[-1].split("-")

    raise ValueError(f"Invalid model path type: {type(model_path)}")


__all__ = [
    "get_model_category",
    "get_model_class",
    "get_model_name_parts",
    "is_valid_module_name",
]
