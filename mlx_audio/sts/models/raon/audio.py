"""RAON's per-frame microphone resampling."""

import numpy as np


def resample_input_frame(audio: np.ndarray) -> np.ndarray:
    """Match the publisher's 24-to-16 kHz Hann sinc filter and zero padding."""
    audio = np.asarray(audio, dtype=np.float32)
    # torchaudio defaults: width 6, rolloff .99, reduced rate ratio 3:2.
    width = 10
    offsets = np.arange(-width, width + 3, dtype=np.float32) / 3
    phases = -np.arange(2, dtype=np.float32)[:, None] / 2
    t = np.clip((phases + offsets) * 1.98, -6, 6)
    window = np.cos(t * np.pi / 6 / 2) ** 2
    kernel = np.sinc(t) * window * np.float32(1.98 / 3)
    padded = np.pad(audio, (width, width + 3))
    frames = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape[1])[::3]
    return (frames @ kernel.T).reshape(-1)[: (audio.size * 2 + 2) // 3]
