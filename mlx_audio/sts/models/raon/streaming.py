from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.mimi.mimi import MimiStreamingDecoder
from mlx_audio.stt.models.voxtral_realtime.audio import compute_mel_filters
from mlx_audio.stt.models.voxtral_realtime.streaming import (
    StreamingConvStem,
    StreamingEncoder,
)
from mlx_audio.utils import resample_audio

INPUT_SAMPLE_RATE = 24_000
ENCODER_SAMPLE_RATE = 16_000
FRAME_RATE = 12.5
INPUT_SAMPLES_PER_FRAME = int(INPUT_SAMPLE_RATE / FRAME_RATE)
ENCODER_SAMPLES_PER_FRAME = int(ENCODER_SAMPLE_RATE / FRAME_RATE)


def prepare_duplex_prompt(tokenizer: Any, system_prompt: str) -> mx.array:
    """Render one source-compatible system prompt for duplex decoding."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer.encode(rendered, add_special_tokens=False)
    return mx.array(tokens, dtype=mx.int32)


class RaonCausalMelStream:
    """Causal Voxtral log-mel frontend used by one Raon duplex session."""

    def __init__(
        self,
        *,
        num_mel_bins: int = 128,
        window_size: int = 400,
        hop_length: int = 160,
        sample_rate: int = ENCODER_SAMPLE_RATE,
    ) -> None:
        self.window_size = window_size
        self.hop_length = hop_length
        filters = compute_mel_filters(
            num_mel_bins=num_mel_bins,
            window_size=window_size,
            sample_rate=sample_rate,
        )
        self._mel_filters = mx.array(filters, dtype=mx.float32)
        n = mx.arange(window_size, dtype=mx.float32)
        self._window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / window_size))
        mx.eval(self._mel_filters, self._window)
        self.reset()

    def reset(self) -> None:
        self._stft_cache = np.zeros(0, dtype=np.float32)
        self._running_max = float("-inf")

    def step(self, samples: np.ndarray) -> mx.array:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        first = self._stft_cache.size == 0
        if first:
            waveform = np.pad(samples, (self.window_size // 2, 0))
        else:
            waveform = np.concatenate([self._stft_cache, samples])

        if waveform.size < self.window_size:
            self._stft_cache = waveform
            return mx.zeros((self._mel_filters.shape[1], 0), dtype=mx.float32)

        num_frames = (waveform.size - self.window_size) // self.hop_length + 1
        if num_frames <= 1:
            self._stft_cache = waveform
            return mx.zeros((self._mel_filters.shape[1], 0), dtype=mx.float32)

        emit_frames = num_frames - 1
        starts = np.arange(emit_frames)[:, None] * self.hop_length
        offsets = np.arange(self.window_size)[None, :]
        frames = mx.array(waveform[starts + offsets], dtype=mx.float32)
        spectrum = mx.fft.rfft(frames * self._window[None, :], axis=-1)
        mel_spec = (mx.abs(spectrum) ** 2) @ self._mel_filters
        log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))

        normalized = []
        for index in range(emit_frames):
            frame = log_spec[index]
            frame_max = float(mx.max(frame).item())
            self._running_max = max(self._running_max, frame_max)
            normalized.append(mx.maximum(frame, self._running_max - 8.0))

        leftover_start = emit_frames * self.hop_length
        self._stft_cache = waveform[leftover_start:].copy()
        output = (mx.stack(normalized, axis=0) + 4.0) / 4.0
        mx.eval(output)
        return output.T


class RaonStreamingAudioEncoder:
    """Turn raw 24 kHz duplex frames into adapted Raon thinker embeddings."""

    def __init__(self, encoder: Any, adaptor: Optional[Callable] = None) -> None:
        self.encoder = encoder
        self.adaptor = adaptor
        self.input_sample_rate = INPUT_SAMPLE_RATE
        self.encoder_sample_rate = ENCODER_SAMPLE_RATE
        self.samples_per_frame = INPUT_SAMPLES_PER_FRAME
        self.reset()

    def reset(self) -> None:
        self._mel = RaonCausalMelStream()
        self._conv = StreamingConvStem(self.encoder.encoder)
        self._transformer = StreamingEncoder(self.encoder.encoder)
        self._encoded_buffer: Optional[mx.array] = None

    def step(self, audio_frame: Any) -> tuple[mx.array, mx.array]:
        audio = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
        if audio.size != self.samples_per_frame:
            raise ValueError(
                "Raon duplex audio frames must contain exactly "
                f"{self.samples_per_frame} samples, got {audio.size}."
            )
        if not np.isfinite(audio).all():
            raise ValueError("Raon duplex audio frames must contain finite samples.")

        encoder_audio = np.asarray(
            resample_audio(
                audio,
                self.input_sample_rate,
                self.encoder_sample_rate,
            ),
            dtype=np.float32,
        )
        if encoder_audio.size != ENCODER_SAMPLES_PER_FRAME:
            raise ValueError(
                "Raon duplex resampling must produce exactly "
                f"{ENCODER_SAMPLES_PER_FRAME} samples, got {encoder_audio.size}."
            )

        mel = self._mel.step(encoder_audio)
        conv = self._conv.step(mel)
        encoded = self._transformer.step(conv)
        if self._encoded_buffer is not None and self._encoded_buffer.shape[0]:
            encoded = mx.concatenate([self._encoded_buffer, encoded], axis=0)

        factor = int(self.encoder.config.downsample_factor)
        usable = (encoded.shape[0] // factor) * factor
        if usable == 0:
            self._encoded_buffer = encoded
            output_size = int(self.encoder.config.stacked_output_size)
            if self.adaptor is not None:
                output_size = int(self.adaptor.linear_fc2.weight.shape[0])
            return (
                mx.zeros((1, 0, output_size), dtype=encoded.dtype),
                mx.zeros((1, 0), dtype=mx.bool_),
            )

        current = encoded[:usable].reshape(
            1,
            -1,
            self.encoder.config.stacked_output_size,
        )
        self._encoded_buffer = encoded[usable:]
        if self.adaptor is not None:
            current = self.adaptor(current)
        mask = mx.ones(current.shape[:2], dtype=mx.bool_)
        mx.eval(current, mask)
        return current, mask


class RaonStreamingAudioDecoder:
    """Decode one Raon acoustic-code frame while retaining Mimi state."""

    def __init__(self, codec: Any) -> None:
        self._decoder = MimiStreamingDecoder(codec)

    def reset(self) -> None:
        self._decoder.reset()

    def decode_frame(self, codes: mx.array) -> mx.array:
        if codes.ndim == 1:
            codes = codes[None, :]
        if codes.ndim != 2 or codes.shape[0] != 1:
            raise ValueError(
                "Raon duplex decoder codes must have shape (1, codebooks)."
            )
        audio = self._decoder.decode_frames(codes[:, :, None])
        mx.eval(audio)
        return audio


@dataclass(frozen=True)
class RaonDuplexAudioFrameResult:
    frame_index: int
    state: Any
    frame_tokens: list[int]
    input_embedding_count: int
    emitted_audio: bool
    audio_codes: mx.array
    audio: mx.array


class RaonDuplexSession:
    """Stateful raw-audio-to-PCM composition for one Raon duplex exchange."""

    def __init__(
        self,
        model: Any,
        system_tokens: mx.array,
        *,
        audio_encoder: Optional[Any] = None,
        audio_decoder: Optional[Any] = None,
        silence_codes: Optional[mx.array] = None,
        speak_first: bool = False,
        text_sampler: Optional[Callable] = None,
        first_code_sampler: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.audio_encoder = audio_encoder or RaonStreamingAudioEncoder(
            model.speech.audio_encoder,
            model.input_adaptor,
        )
        self.audio_decoder = audio_decoder or RaonStreamingAudioDecoder(model.codec)
        self.text_sampler = text_sampler
        self.first_code_sampler = first_code_sampler
        self.samples_per_frame = int(model.sample_rate / FRAME_RATE)
        if self.samples_per_frame != INPUT_SAMPLES_PER_FRAME:
            raise ValueError(
                "Raon SpeechChat requires 24 kHz audio at 12.5 frames per second."
            )
        self.silence_codes = (
            silence_codes if silence_codes is not None else self._encode_silence()
        )
        self.state = model.init_duplex_state(
            system_tokens,
            speak_first=speak_first,
            text_sampler=text_sampler,
            first_code_sampler=first_code_sampler,
        )
        self.frame_index = 0

    def _encode_silence(self) -> mx.array:
        silence = mx.zeros((1, 1, self.samples_per_frame), dtype=mx.float32)
        codes = self.model.codec.encode(silence)
        groups = int(self.model.config.components.code_predictor.num_code_groups)
        if codes.ndim != 3 or codes.shape[0] != 1 or codes.shape[-1] < 1:
            raise ValueError("Raon Mimi silence encoding returned an invalid shape.")
        silence_codes = codes[:, :groups, 0]
        mx.eval(silence_codes)
        return silence_codes

    def step(self, audio_frame: Any) -> RaonDuplexAudioFrameResult:
        audio = np.asarray(audio_frame, dtype=np.float32).reshape(-1)
        if audio.size != self.samples_per_frame:
            raise ValueError(
                "Raon duplex audio frames must contain exactly "
                f"{self.samples_per_frame} samples, got {audio.size}."
            )
        embeddings, mask = self.audio_encoder.step(audio)
        frame = self.model.duplex_frame(
            self.state,
            audio_input_embeds=embeddings,
            audio_input_embeds_mask=mask,
            text_sampler=self.text_sampler,
            first_code_sampler=self.first_code_sampler,
        )
        codes = (
            frame.emitted_codes
            if frame.emitted_codes is not None
            else self.silence_codes
        )
        decoded = self.audio_decoder.decode_frame(codes)
        if decoded.ndim != 3 or decoded.shape[:2] != (1, 1):
            raise ValueError(
                "Raon duplex Mimi decode must return shape (1, 1, samples), "
                f"got {decoded.shape}."
            )
        if decoded.shape[-1] != self.samples_per_frame:
            raise ValueError(
                "Raon duplex Mimi decode must return exactly "
                f"{self.samples_per_frame} samples, got {decoded.shape[-1]}."
            )
        result = RaonDuplexAudioFrameResult(
            frame_index=self.frame_index,
            state=frame.state,
            frame_tokens=frame.frame_tokens,
            input_embedding_count=int(embeddings.shape[1]),
            emitted_audio=frame.emitted_audio,
            audio_codes=codes,
            audio=decoded[0, 0],
        )
        self.state = frame.state
        self.frame_index += 1
        return result

    def process(self, audio_input: Any) -> Iterator[RaonDuplexAudioFrameResult]:
        audio = np.asarray(audio_input, dtype=np.float32)
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        if audio.ndim != 1:
            raise ValueError("Raon fixed duplex input must be mono audio.")
        complete = (audio.size // self.samples_per_frame) * self.samples_per_frame
        for start in range(0, complete, self.samples_per_frame):
            yield self.step(audio[start : start + self.samples_per_frame])
