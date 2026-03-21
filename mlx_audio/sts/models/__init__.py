# Copyright (c) 2025 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

import importlib

__all__ = [
    "SAMAudio",
    "SAMAudioProcessor",
    "SeparationResult",
    "Batch",
    "save_audio",
    "SAMAudioConfig",
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
    # LFM2.5-Audio
    "LFM2AudioModel",
    "LFM2AudioProcessor",
    "LFM2AudioConfig",
    "LFMModality",
    "ChatState",
    "GenerationConfig",
]

_EXPORTS = {
    "DeepFilterNetModel": ("deepfilternet", "DeepFilterNetModel"),
    "DeepFilterNetConfig": ("deepfilternet", "DeepFilterNetConfig"),
    "DeepFilterNet2Config": ("deepfilternet", "DeepFilterNet2Config"),
    "DeepFilterNet3Config": ("deepfilternet", "DeepFilterNet3Config"),
    "DeepFilterNetStreamer": ("deepfilternet", "DeepFilterNetStreamer"),
    "DeepFilterNetStreamingConfig": ("deepfilternet", "DeepFilterNetStreamingConfig"),
    "MossFormer2SE": ("mossformer2_se", "MossFormer2SE"),
    "MossFormer2SEConfig": ("mossformer2_se", "MossFormer2SEConfig"),
    "MossFormer2SEModel": ("mossformer2_se", "MossFormer2SEModel"),
    "SAMAudio": ("sam_audio", "SAMAudio"),
    "SAMAudioProcessor": ("sam_audio", "SAMAudioProcessor"),
    "SeparationResult": ("sam_audio", "SeparationResult"),
    "Batch": ("sam_audio", "Batch"),
    "save_audio": ("sam_audio", "save_audio"),
    "SAMAudioConfig": ("sam_audio", "SAMAudioConfig"),
    "LFM2AudioModel": ("lfm_audio", "LFM2AudioModel"),
    "LFM2AudioProcessor": ("lfm_audio", "LFM2AudioProcessor"),
    "LFM2AudioConfig": ("lfm_audio", "LFM2AudioConfig"),
    "LFMModality": ("lfm_audio", "LFMModality"),
    "ChatState": ("lfm_audio", "ChatState"),
    "GenerationConfig": ("lfm_audio", "GenerationConfig"),
}

_SUBMODULES = {
    "deepfilternet",
    "sam_audio",
    "lfm_audio",
    "mossformer2_se",
}


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")

    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(module, attr_name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _SUBMODULES | set(__all__))
