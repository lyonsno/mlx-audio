import importlib

__all__ = [
    "SAMAudio",
    "SAMAudioProcessor",
    "SeparationResult",
    "Batch",
    "save_audio",
    "SAMAudioConfig",
    "VoicePipeline",
    # DeepFilterNet
    "DeepFilterNetModel",
    "DeepFilterNetConfig",
    "DeepFilterNet2Config",
    "DeepFilterNet3Config",
    "DeepFilterNetStreamer",
    "DeepFilterNetStreamingConfig",
    # MossFormer2 SE
    "MossFormer2SE",
    "MossFormer2SEConfig",
    "MossFormer2SEModel",
]

_EXPORTS = {
    "SAMAudio": ("models.sam_audio", "SAMAudio"),
    "SAMAudioProcessor": ("models.sam_audio", "SAMAudioProcessor"),
    "SeparationResult": ("models.sam_audio", "SeparationResult"),
    "Batch": ("models.sam_audio", "Batch"),
    "save_audio": ("models.sam_audio", "save_audio"),
    "SAMAudioConfig": ("models.sam_audio", "SAMAudioConfig"),
    "VoicePipeline": ("voice_pipeline", "VoicePipeline"),
    "DeepFilterNetModel": ("models.deepfilternet", "DeepFilterNetModel"),
    "DeepFilterNetConfig": ("models.deepfilternet", "DeepFilterNetConfig"),
    "DeepFilterNet2Config": ("models.deepfilternet", "DeepFilterNet2Config"),
    "DeepFilterNet3Config": ("models.deepfilternet", "DeepFilterNet3Config"),
    "DeepFilterNetStreamer": ("models.deepfilternet", "DeepFilterNetStreamer"),
    "DeepFilterNetStreamingConfig": (
        "models.deepfilternet",
        "DeepFilterNetStreamingConfig",
    ),
    "MossFormer2SE": ("models.mossformer2_se", "MossFormer2SE"),
    "MossFormer2SEConfig": ("models.mossformer2_se", "MossFormer2SEConfig"),
    "MossFormer2SEModel": ("models.mossformer2_se", "MossFormer2SEModel"),
}

_OPTIONAL_VOICE_PIPELINE_IMPORTS = (
    "sounddevice",
    "webrtcvad",
    "mlx_lm",
    "pkg_resources",
)


def _is_missing_voice_pipeline_optional_dep(exc: ImportError) -> bool:
    missing_name = getattr(exc, "name", None)
    if not missing_name:
        return False

    normalized_name = missing_name.lstrip("_")
    return any(
        normalized_name == dep or normalized_name.startswith(f"{dep}.")
        for dep in _OPTIONAL_VOICE_PIPELINE_IMPORTS
    )


def __getattr__(name):
    if name == "models":
        return importlib.import_module(f"{__name__}.models")

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        try:
            module = importlib.import_module(f"{__name__}.{module_name}")
        except ImportError as exc:
            if name == "VoicePipeline" and _is_missing_voice_pipeline_optional_dep(exc):
                # Preserve the historical contract: optional realtime deps map
                # to a sentinel None instead of raising on import.
                return None
            raise
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
