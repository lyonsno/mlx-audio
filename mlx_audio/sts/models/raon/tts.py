import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Union,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3ModelArgs
from mlx_audio.lm.models.qwen3 import Qwen3Model
from mlx_audio.lm.sample_utils import apply_top_p
from mlx_audio.tts.models.base import GenerationResult
from mlx_audio.utils import get_model_path, load_weights

from .components import RaonComponentConfig, RaonSpeechComponents
from .duplex import (
    AUDIO_INPUT_PLACEHOLDER_ID,
    AUDIO_OUTPUT_BACKCHANNEL_ID,
    AUDIO_OUTPUT_END_PAD_ID,
    AUDIO_OUTPUT_PAD_ID,
    AUDIO_OUTPUT_PLACEHOLDER_ID,
    AUDIO_OUTPUT_SIL_ID,
    AUDIO_START_ID,
    DuplexMachineState,
    DuplexStateConfig,
    DuplexStateManager,
)

AUDIO_END_ID = 151670
IM_START_ID = 151644
IM_END_ID = 151645


@dataclass(frozen=True)
class RaonEmbeddingAdaptorConfig:
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    output_time_scale: float
    use_post_norm: bool
    norm_eps: float
    post_norm_init_scale: float

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonEmbeddingAdaptorConfig":
        config = cls(
            input_size=int(values["input_size"]),
            output_size=int(values["output_size"]),
            hidden_size=int(values.get("hidden_size") or values["output_size"]),
            num_layers=int(values.get("num_layers", 1)),
            output_time_scale=float(values.get("output_time_scale", 1)),
            use_post_norm=bool(values.get("use_post_norm", False)),
            norm_eps=float(values.get("norm_eps", 1e-6)),
            post_norm_init_scale=float(values.get("post_norm_init_scale", 1.0)),
        )
        if config.num_layers != 2 or config.output_time_scale != 1:
            raise ValueError(
                "Raon TTS currently requires a two-layer output adaptor with "
                "output_time_scale=1."
            )
        return config


RaonOutputAdaptorConfig = RaonEmbeddingAdaptorConfig


@dataclass(frozen=True)
class RaonTTSConfig:
    thinker: Qwen3ModelArgs
    components: RaonComponentConfig
    output_adaptor: RaonOutputAdaptorConfig
    num_quantizers: int
    codebook_size: int

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonTTSConfig":
        audio_tokenizer = values["audio_tokenizer_config"]
        codebook_size = int(audio_tokenizer["codebook_size"])
        components = RaonComponentConfig.from_dict(values)
        thinker = Qwen3ModelArgs.from_dict(values["text_model_config"])
        output_adaptor = RaonOutputAdaptorConfig.from_dict(
            values["output_adaptor_config"]
        )
        if output_adaptor.output_size != thinker.hidden_size:
            raise ValueError(
                "Raon output adaptor size must match the thinker hidden size: "
                f"{output_adaptor.output_size} != {thinker.hidden_size}."
            )
        return cls(
            thinker=thinker,
            components=components,
            output_adaptor=output_adaptor,
            num_quantizers=int(audio_tokenizer.get("num_quantizers", 32)),
            codebook_size=codebook_size,
        )


@dataclass(frozen=True)
class RaonSpeechConfig(RaonTTSConfig):
    input_adaptor: RaonEmbeddingAdaptorConfig

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonSpeechConfig":
        base = RaonTTSConfig.from_dict(values)
        input_adaptor = RaonEmbeddingAdaptorConfig.from_dict(
            values["input_adaptor_config"]
        )
        if (
            input_adaptor.input_size
            != base.components.audio_encoder.stacked_output_size
        ):
            raise ValueError(
                "Raon input adaptor size must match the Voxtral stacked output: "
                f"{input_adaptor.input_size} != "
                f"{base.components.audio_encoder.stacked_output_size}."
            )
        if input_adaptor.output_size != base.thinker.hidden_size:
            raise ValueError(
                "Raon input adaptor output must match the thinker hidden size: "
                f"{input_adaptor.output_size} != {base.thinker.hidden_size}."
            )
        return cls(
            thinker=base.thinker,
            components=base.components,
            output_adaptor=base.output_adaptor,
            num_quantizers=base.num_quantizers,
            codebook_size=base.codebook_size,
            input_adaptor=input_adaptor,
        )


@dataclass(frozen=True)
class RaonDuplexConfig(RaonSpeechConfig):
    state: DuplexStateConfig

    @classmethod
    def from_speech_config(
        cls,
        config: RaonSpeechConfig,
        *,
        state: DuplexStateConfig,
    ) -> "RaonDuplexConfig":
        return cls(
            thinker=config.thinker,
            components=config.components,
            output_adaptor=config.output_adaptor,
            num_quantizers=config.num_quantizers,
            codebook_size=config.codebook_size,
            input_adaptor=config.input_adaptor,
            state=state,
        )

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonDuplexConfig":
        speech = RaonSpeechConfig.from_dict(values)
        state = DuplexStateConfig(
            use_duplex_end_pad=bool(values.get("use_duplex_end_pad", False)),
            use_sil_token=bool(values.get("use_sil_token", False)),
            no_audio_in_sil=bool(values.get("no_audio_in_sil", False)),
            sequence_mode=values.get("sequence_mode"),
            duplex_pad_token_id=int(
                values.get("duplex_pad_token_id", AUDIO_OUTPUT_PAD_ID)
            ),
            duplex_end_pad_token_id=int(
                values.get("duplex_end_pad_token_id", AUDIO_OUTPUT_END_PAD_ID)
            ),
            duplex_sil_token_id=int(
                values.get("duplex_sil_token_id", AUDIO_OUTPUT_SIL_ID)
            ),
            use_backchannel_token=bool(values.get("use_backchannel_token", False)),
            duplex_bc_token_id=int(
                values.get("duplex_bc_token_id", AUDIO_OUTPUT_BACKCHANNEL_ID)
            ),
        )
        return cls.from_speech_config(speech, state=state)


class RaonEmbeddingAdaptor(nn.Module):
    def __init__(self, config: RaonEmbeddingAdaptorConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(config.hidden_size, config.output_size, bias=False)
        self.post_norm = (
            nn.RMSNorm(config.output_size, eps=config.norm_eps)
            if config.use_post_norm
            else None
        )
        if self.post_norm is not None:
            self.post_norm.weight = mx.full(
                (config.output_size,), config.post_norm_init_scale
            )

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs = self.linear_fc2(nn.gelu(self.linear_fc1(inputs)))
        if self.post_norm is not None:
            outputs = self.post_norm(outputs)
        return outputs


RaonOutputAdaptor = RaonEmbeddingAdaptor


class RaonThinker(nn.Module):
    """Qwen3 thinker returning both pre-norm and normalized final states."""

    def __init__(self, config: Qwen3ModelArgs):
        super().__init__()
        model = Qwen3Model(config)
        self.embed_tokens = model.embed_tokens
        self.layers = model.layers
        self.norm = model.norm

    def __call__(
        self,
        input_ids: mx.array,
        *,
        cache: Optional[list[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        hidden = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            hidden = layer(hidden, mask, layer_cache)
        return hidden, self.norm(hidden)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


@dataclass(frozen=True)
class RaonTTSStep:
    index: int
    first_code: int
    thinker_cache_offset: int
    talker_cache_offset: int
    used_previous_audio: bool


@dataclass(frozen=True)
class RaonTTSResult:
    audio_codes: mx.array
    audio: mx.array
    finish_reason: Literal["audio_end", "length"]
    steps: tuple[RaonTTSStep, ...]


@dataclass(frozen=True)
class RaonDuplexFrameState:
    sequences: mx.array
    audio_codes: mx.array
    thinker_cache: list[KVCache]
    talker_cache: list[KVCache]
    machine_state: DuplexMachineState
    forced_sil_remaining: int = 0


@dataclass(frozen=True)
class RaonDuplexFrameResult:
    state: RaonDuplexFrameState
    frame_tokens: list[int]
    emitted_audio: bool
    emitted_codes: Optional[mx.array]


def prepare_tts_prompt(tokenizer: Any, text: str) -> mx.array:
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": f"Speak the following text:\n{text}",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    if len(im_end_ids) != 1:
        raise ValueError("Raon tokenizer must encode <|im_end|> as one token.")
    trailing = {AUDIO_END_ID, int(im_end_ids[0])}
    while input_ids and input_ids[-1] in trailing:
        input_ids.pop()
    input_ids.append(AUDIO_START_ID)
    return mx.array(input_ids, dtype=mx.int32)


def _apply_source_top_k(logits: mx.array, top_k: int) -> mx.array:
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    cutoff = mx.sort(logits, axis=-1)[..., -top_k]
    return mx.where(logits < cutoff[..., None], -mx.inf, logits)


def _sample_logits(
    logits: mx.array,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mx.array:
    scores = logits.astype(mx.float32) / temperature
    if top_k:
        scores = _apply_source_top_k(scores, top_k)
    logprobs = scores - mx.logsumexp(scores, axis=-1, keepdims=True)
    if top_p < 1:
        logprobs = apply_top_p(logprobs, top_p)
    return mx.random.categorical(logprobs, axis=-1)


def make_first_code_sampler(
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    ras_enabled: bool,
    ras_window_size: int,
    ras_repetition_threshold: float,
    audio_end_code: int,
) -> Callable[[mx.array], mx.array]:
    """Build the source TTS sampler for the first codec group."""
    if temperature <= 0:
        raise ValueError("Raon TTS sampling temperature must be positive.")
    if top_k < 0:
        raise ValueError("Raon TTS top_k must be non-negative.")
    if not 0 < top_p <= 1:
        raise ValueError("Raon TTS top_p must be in the interval (0, 1].")
    if ras_window_size <= 0:
        raise ValueError("Raon TTS RAS window size must be positive.")
    if not 0 <= ras_repetition_threshold <= 1:
        raise ValueError("Raon TTS RAS repetition threshold must be in [0, 1].")

    history: list[int] = []

    def sample_first_code(logits: mx.array) -> mx.array:
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(
                "Raon TTS first-code sampling currently requires batch size 1."
            )
        effective_top_k = 0 if top_k >= logits.shape[-1] else top_k
        sampled = _sample_logits(
            logits,
            temperature=temperature,
            top_k=effective_top_k,
            top_p=top_p,
        )
        sampled_token = int(sampled.item())
        window = history[-ras_window_size:]
        if ras_enabled and window:
            repetition_ratio = sum(token == sampled_token for token in window) / len(
                window
            )
            if repetition_ratio > ras_repetition_threshold:
                sampled = _sample_logits(
                    logits,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                )
                sampled_token = int(sampled.item())
        if sampled_token != audio_end_code:
            history.append(sampled_token)
        return mx.array([sampled_token], dtype=mx.int32)

    return sample_first_code


class RaonTTSModel(nn.Module):
    _REQUIRED_SOURCE_FAMILIES = (
        "text_model.",
        "lm_head.",
        "output_adaptor.",
        "audio_tokenizer.",
        "talker.",
        "thinker_to_talker_proj.",
        "audio_lm_head.",
        "proj_code.",
        "code_predictor.",
    )
    _EXCLUDED_SOURCE_FAMILIES = {
        "audio_encoder.": "TTS-only loading excludes the speech-input encoder.",
        "input_adaptor.": "TTS-only loading excludes the speech-input adaptor.",
        "speaker_encoder.": "Speaker conditioning is not implemented by this TTS path.",
    }
    supports_streaming = False
    supports_voice = False
    supports_temperature = True
    _SUPPORTS_AUDIO_INPUT = False

    def __init__(self, config: RaonTTSConfig, codec: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.thinker = RaonThinker(config.thinker)
        self.lm_head = nn.Linear(
            config.thinker.hidden_size, config.thinker.vocab_size, bias=False
        )
        self.output_adaptor = RaonOutputAdaptor(config.output_adaptor)
        self.speech = RaonSpeechComponents(config.components)
        if not self._SUPPORTS_AUDIO_INPUT:
            self.speech.audio_encoder = None
        self.codec = codec or Mimi(
            mimi_202407(
                config.num_quantizers,
                transformers_compatible=True,
            )
        )
        self.tokenizer = None
        self.max_frames = 512
        self.weight_admission: Optional[Dict[str, Any]] = None

    @property
    def sample_rate(self) -> int:
        return int(self.codec.sample_rate)

    def feedback_embedding(self, audio_codes: mx.array) -> mx.array:
        if audio_codes.ndim != 3:
            raise ValueError(
                "Raon audio codes must have shape (batch, frames, groups)."
            )
        latent = self.codec.quantizer.decode(audio_codes.transpose(0, 2, 1))
        return self.output_adaptor(latent.transpose(0, 2, 1))

    def _forward(
        self,
        input_ids: mx.array,
        thinker_cache: list[KVCache],
        talker_cache: list[KVCache],
        input_embeddings: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        accepted, normalized = self.thinker(
            input_ids,
            cache=thinker_cache,
            input_embeddings=input_embeddings,
        )
        text_logits = self.lm_head(normalized)
        talker_inputs = self.speech.thinker_to_talker_proj(accepted)
        talker_hidden = self.speech.talker(talker_inputs, cache=talker_cache)
        return talker_hidden, text_logits

    def generate_frames(
        self,
        input_ids: mx.array,
        *,
        max_frames: int,
        first_code_sampler: Optional[Callable[[mx.array], Any]] = None,
    ) -> RaonTTSResult:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                "Raon TTS currently accepts one tokenized prompt at a time."
            )
        if int(input_ids[0, -1].item()) != AUDIO_START_ID:
            raise ValueError("Raon TTS prompts must terminate with AUDIO_START_ID.")

        thinker_cache = self.thinker.make_cache()
        talker_cache = self.speech.talker.make_cache()
        talker_hidden, _ = self._forward(input_ids, thinker_cache, talker_cache)
        frames = []
        steps = []
        finish_reason: Literal["audio_end", "length"] = "length"
        used_previous_audio = False

        for index in range(max_frames + 1):
            codes = self.speech.generate_audio_codes_from_talker_hidden(
                talker_hidden,
                first_code_sampler=first_code_sampler,
            )
            mx.eval(codes)
            first_code = int(codes[0, 0].item())
            steps.append(
                RaonTTSStep(
                    index=index,
                    first_code=first_code,
                    thinker_cache_offset=thinker_cache[0].offset,
                    talker_cache_offset=talker_cache[0].offset,
                    used_previous_audio=used_previous_audio,
                )
            )
            if first_code == self.config.codebook_size:
                finish_reason = "audio_end"
                break
            if len(frames) == max_frames:
                break
            frames.append(codes)
            if len(frames) == max_frames:
                break

            previous = codes[:, None, :]
            feedback = self.feedback_embedding(previous)
            placeholder = mx.full((1, 1), AUDIO_OUTPUT_PLACEHOLDER_ID, dtype=mx.int32)
            talker_hidden, _ = self._forward(
                placeholder,
                thinker_cache,
                talker_cache,
                input_embeddings=feedback,
            )
            used_previous_audio = True

        if frames:
            audio_codes = mx.stack(frames, axis=1)
            audio = self.codec.decode(audio_codes.transpose(0, 2, 1))
        else:
            audio_codes = mx.zeros(
                (1, 0, self.config.components.code_predictor.num_code_groups),
                dtype=mx.int32,
            )
            audio = mx.zeros((1, 1, 0), dtype=mx.float32)
        return RaonTTSResult(
            audio_codes=audio_codes,
            audio=audio,
            finish_reason=finish_reason,
            steps=tuple(steps),
        )

    def generate_text_frames(
        self,
        text: str,
        *,
        max_frames: int,
        temperature: float = 1.2,
        top_k: int = 20,
        top_p: float = 0.8,
        ras_enabled: bool = True,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
    ) -> RaonTTSResult:
        if self.tokenizer is None:
            raise RuntimeError("Raon tokenizer is not loaded.")
        return self.generate_frames(
            prepare_tts_prompt(self.tokenizer, text),
            max_frames=max_frames,
            first_code_sampler=make_first_code_sampler(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                ras_enabled=ras_enabled,
                ras_window_size=ras_window_size,
                ras_repetition_threshold=ras_repetition_threshold,
                audio_end_code=self.config.codebook_size,
            ),
        )

    def _finalize_waveform(self, result: RaonTTSResult) -> mx.array:
        raw = result.audio
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 1:
            raise ValueError(
                "Raon Mimi decode must return shape (1, 1, samples), "
                f"got {raw.shape}."
            )
        samples_per_frame = int(self.sample_rate / self.codec.frame_rate)
        valid_samples = int(result.audio_codes.shape[1]) * samples_per_frame
        waveform = raw[0, 0, : min(valid_samples, int(raw.shape[-1]))]
        if waveform.shape[0] <= samples_per_frame:
            return waveform[:0]
        return waveform[:-samples_per_frame]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        milliseconds = int(seconds * 1000)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        whole_seconds, milliseconds = divmod(milliseconds, 1000)
        return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = 20,
        top_p: Optional[float] = 0.8,
        ras_enabled: bool = True,
        ras_window_size: int = 50,
        ras_repetition_threshold: float = 0.5,
        stream: bool = False,
        streaming_interval: float = 2.0,
        max_frames: Optional[int] = None,
        **_: Any,
    ) -> Iterator[GenerationResult]:
        """Generate one source-finalized result through MLX Audio's TTS protocol."""
        if voice is not None:
            raise ValueError("Raon TTS does not yet support voice conditioning.")
        if stream:
            raise ValueError(
                "Raon TTS does not yet support incremental audio streaming."
            )
        _ = streaming_interval
        if self.tokenizer is None:
            raise RuntimeError("Raon tokenizer is not loaded.")

        effective_max_frames = self.max_frames if max_frames is None else max_frames
        prompt = prepare_tts_prompt(self.tokenizer, text)
        first_code_sampler = make_first_code_sampler(
            temperature=1.2 if temperature is None else temperature,
            top_k=20 if top_k is None else top_k,
            top_p=0.8 if top_p is None else top_p,
            ras_enabled=ras_enabled,
            ras_window_size=ras_window_size,
            ras_repetition_threshold=ras_repetition_threshold,
            audio_end_code=self.config.codebook_size,
        )
        started = time.perf_counter()
        result = self.generate_frames(
            prompt,
            max_frames=effective_max_frames,
            first_code_sampler=first_code_sampler,
        )
        waveform = self._finalize_waveform(result)
        mx.eval(waveform)
        elapsed = time.perf_counter() - started
        samples = int(waveform.shape[0])
        duration_seconds = samples / self.sample_rate if self.sample_rate else 0.0
        generated_frames = int(result.audio_codes.shape[1])
        yield GenerationResult(
            audio=waveform,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=generated_frames,
            audio_duration=self._format_duration(duration_seconds),
            real_time_factor=(duration_seconds / elapsed if elapsed > 0 else 0.0),
            prompt={
                "text": text,
                "tokens": int(prompt.shape[0]),
                "finish_reason": result.finish_reason,
                "first_codes": [step.first_code for step in result.steps],
                "thinker_cache_offsets": [
                    step.thinker_cache_offset for step in result.steps
                ],
                "talker_cache_offsets": [
                    step.talker_cache_offset for step in result.steps
                ],
            },
            audio_samples={
                "samples": samples,
                "frames": generated_frames,
                "samples-per-sec": (
                    round(samples / elapsed, 2) if elapsed > 0 else 0.0
                ),
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
            is_streaming_chunk=False,
            is_final_chunk=True,
        )

    def validate_source_families(self, weights: Dict[str, mx.array]) -> None:
        missing = [
            prefix
            for prefix in self._REQUIRED_SOURCE_FAMILIES
            if not any(name.startswith(prefix) for name in weights)
        ]
        if missing:
            raise ValueError(f"Raon TTS missing source families: {missing}.")

    def classify_source_weights(self, weights: Dict[str, mx.array]) -> Dict[str, Any]:
        excluded_names: Dict[str, list[str]] = {
            prefix: [] for prefix in self._EXCLUDED_SOURCE_FAMILIES
        }
        supported: Dict[str, mx.array] = {}
        unknown_roots = []
        for name, value in weights.items():
            excluded_prefix = next(
                (
                    prefix
                    for prefix in self._EXCLUDED_SOURCE_FAMILIES
                    if name.startswith(prefix)
                ),
                None,
            )
            if excluded_prefix is not None:
                excluded_names[excluded_prefix].append(name)
            elif name.startswith(self._REQUIRED_SOURCE_FAMILIES):
                supported[name] = value
            else:
                unknown_roots.append(name)
        if unknown_roots:
            raise ValueError(
                "Raon TTS has unclassified source tensors: " f"{sorted(unknown_roots)}."
            )

        mapped = self.sanitize(supported)
        qkv_groups: Dict[str, set[str]] = {}
        for name in supported:
            if not name.startswith("audio_tokenizer."):
                continue
            for projection in ("q_proj", "k_proj", "v_proj"):
                suffix = f".{projection}.weight"
                if name.endswith(suffix):
                    group = name.removesuffix(suffix)
                    qkv_groups.setdefault(group, set()).add(projection)
                    break
        incomplete_qkv = {
            group: sorted(projections)
            for group, projections in qkv_groups.items()
            if projections != {"q_proj", "k_proj", "v_proj"}
        }
        if incomplete_qkv:
            raise ValueError(
                f"Raon TTS incomplete Mimi QKV source groups: {incomplete_qkv}."
            )
        fused_source_reduction = 2 * len(qkv_groups)
        expected_target_count = len(supported) - fused_source_reduction
        if len(mapped) != expected_target_count:
            raise ValueError(
                "Raon TTS has unclassified source tensors inside supported families: "
                f"supported={len(supported)}, fused_reduction={fused_source_reduction}, "
                f"mapped={len(mapped)}, expected_mapped={expected_target_count}."
            )

        excluded_counts = {
            prefix: len(names) for prefix, names in excluded_names.items() if names
        }
        return {
            "source_tensor_count": len(weights),
            "mapped_source_tensor_count": len(supported),
            "admitted_target_count": len(mapped),
            "fused_source_reduction": fused_source_reduction,
            "excluded_source_count": sum(excluded_counts.values()),
            "excluded_families": excluded_counts,
            "excluded_reasons": {
                prefix: self._EXCLUDED_SOURCE_FAMILIES[prefix]
                for prefix in excluded_counts
            },
            "excluded_source_names": {
                prefix: sorted(names)
                for prefix, names in excluded_names.items()
                if names
            },
            "unclassified_source_count": 0,
        }

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        mapped: Dict[str, mx.array] = {}
        for source_name, value in weights.items():
            if source_name.startswith("text_model."):
                mapped[f"thinker.{source_name.removeprefix('text_model.')}"] = value
            elif source_name == "lm_head.weight":
                mapped[source_name] = value
            elif source_name == "output_adaptor.proj.0.weight":
                mapped["output_adaptor.linear_fc1.weight"] = value
            elif source_name == "output_adaptor.proj.2.weight":
                mapped["output_adaptor.linear_fc2.weight"] = value
            elif source_name == "output_adaptor.post_norm.weight":
                mapped[source_name] = value

        speech_source = {
            name: value
            for name, value in weights.items()
            if name.startswith(
                (
                    "talker.",
                    "code_predictor.",
                    "thinker_to_talker_proj.",
                    "audio_lm_head.",
                    "proj_code.",
                )
            )
        }
        mapped.update(
            {
                f"speech.{name}": value
                for name, value in RaonSpeechComponents.sanitize(speech_source).items()
            }
        )
        if any(name.startswith("audio_tokenizer.") for name in weights):
            codec = Mimi.sanitize_transformers_weights(
                weights, prefix="audio_tokenizer."
            )
            mapped.update({f"codec.{name}": value for name, value in codec.items()})
        return mapped

    def load_source_weights(self, weights: Dict[str, mx.array]) -> Dict[str, Any]:
        receipt = self.classify_source_weights(weights)
        self.validate_source_families(weights)
        mapped = self.sanitize(weights)
        expected = {name for name, _ in tree_flatten(self.parameters())}
        received = set(mapped)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            raise ValueError(
                "Raon TTS weight admission failed: "
                f"missing={missing}, unexpected={unexpected}."
            )
        self.load_weights(list(mapped.items()), strict=True)
        if isinstance(self.codec, Mimi):
            self.codec._finalize_loaded_weights(self.codec)
        receipt.update({"expected": len(expected), "admitted": len(received)})
        self.weight_admission = receipt
        return receipt

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        *,
        revision: Optional[str] = None,
        weight_files: Optional[Sequence[Union[str, Path, BinaryIO]]] = None,
        weight_format: Optional[str] = None,
    ) -> "RaonTTSModel":
        model_path = get_model_path(path_or_repo, revision=revision)
        with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
            config = cls._load_config(json.load(handle))
        model = cls(config)
        model.load_source_weights(
            load_weights(
                Path(model_path),
                weight_files=weight_files,
                weight_format=weight_format,
            )
        )
        mx.eval(model.parameters())

        from transformers import AutoTokenizer

        model.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model.eval()
        return model

    @staticmethod
    def _load_config(values: Dict[str, Any]) -> RaonTTSConfig:
        return RaonTTSConfig.from_dict(values)


class RaonSpeechModel(RaonTTSModel):
    """Raon speech-input boundary through source-compatible thinker prefill."""

    _REQUIRED_SOURCE_FAMILIES = RaonTTSModel._REQUIRED_SOURCE_FAMILIES + (
        "audio_encoder.",
        "input_adaptor.",
    )
    _EXCLUDED_SOURCE_FAMILIES = {
        "speaker_encoder.": (
            "Speaker conditioning is not implemented by the speech-input path."
        ),
    }
    _SUPPORTS_AUDIO_INPUT = True

    def __init__(self, config: RaonSpeechConfig, codec: Optional[Any] = None):
        super().__init__(config, codec=codec)
        self.config = config
        self.input_adaptor = RaonEmbeddingAdaptor(config.input_adaptor)

    def adapt_audio_embeddings(
        self,
        encoded: mx.array,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        if encoded.ndim != 3:
            raise ValueError(
                "Raon Voxtral embeddings must have shape (batch, frames, dimension)."
            )
        if encoded.shape[-1] != self.config.input_adaptor.input_size:
            raise ValueError(
                "Raon Voxtral embedding size does not match the input adaptor: "
                f"{encoded.shape[-1]} != {self.config.input_adaptor.input_size}."
            )
        if mask is None:
            mask = mx.ones(encoded.shape[:2], dtype=mx.bool_)
        if mask.ndim != 2 or mask.shape != encoded.shape[:2]:
            raise ValueError(
                "Raon audio embedding mask must match the batch and frame axes: "
                f"mask={mask.shape}, embeddings={encoded.shape}."
            )
        return self.input_adaptor(encoded), mask.astype(mx.bool_)

    def get_audio_input_embeds(
        self,
        mel: mx.array,
        mask: Optional[mx.array] = None,
    ) -> tuple[mx.array, mx.array]:
        encoded = self.speech.encode_audio_features(mel)
        return self.adapt_audio_embeddings(encoded, mask)

    def prepare_speech_embeddings(
        self,
        input_ids: mx.array,
        audio_input_embeds: mx.array,
        audio_input_embeds_mask: mx.array,
    ) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError("Raon speech input IDs must have shape (batch, sequence).")
        if audio_input_embeds.ndim != 3:
            raise ValueError(
                "Raon audio input embeddings must have shape "
                "(batch, frames, hidden_size)."
            )
        if audio_input_embeds.shape[0] != input_ids.shape[0]:
            raise ValueError("Raon speech input batch dimensions must match.")
        if audio_input_embeds.shape[-1] != self.config.thinker.hidden_size:
            raise ValueError(
                "Raon audio input embedding size must match the thinker hidden size."
            )
        if (
            audio_input_embeds_mask.ndim != 2
            or audio_input_embeds_mask.shape != audio_input_embeds.shape[:2]
        ):
            raise ValueError(
                "Raon audio input mask must match the batch and frame axes."
            )

        inputs_embeds = self.thinker.embed_tokens(input_ids)
        hidden_size = int(inputs_embeds.shape[-1])
        placeholder_mask_by_row = input_ids == AUDIO_INPUT_PLACEHOLDER_ID
        valid_audio_mask_by_row = audio_input_embeds_mask.astype(mx.bool_)
        placeholder_counts = mx.sum(placeholder_mask_by_row, axis=1)
        valid_audio_counts = mx.sum(valid_audio_mask_by_row, axis=1)
        mx.eval(placeholder_counts, valid_audio_counts)
        for batch_row in range(input_ids.shape[0]):
            row_placeholder_count = int(placeholder_counts[batch_row].item())
            row_valid_audio_count = int(valid_audio_counts[batch_row].item())
            if row_placeholder_count != row_valid_audio_count:
                raise ValueError(
                    "Raon audio placeholder count must match valid audio frames in "
                    f"batch row {batch_row}: placeholders={row_placeholder_count}, "
                    f"valid_frames={row_valid_audio_count}."
                )

        placeholder_mask = placeholder_mask_by_row.reshape(-1)
        valid_audio_mask = valid_audio_mask_by_row.reshape(-1)
        placeholder_count = int(mx.sum(placeholder_mask).item())
        valid_audio_count = int(mx.sum(valid_audio_mask).item())

        flat_inputs = inputs_embeds.reshape(-1, hidden_size)
        flat_audio = audio_input_embeds.reshape(-1, hidden_size)
        input_indices = mx.arange(flat_inputs.shape[0])
        audio_indices = mx.arange(flat_audio.shape[0])
        placeholder_positions = mx.sort(
            mx.where(placeholder_mask, input_indices, flat_inputs.shape[0])
        )[:placeholder_count]
        valid_audio_positions = mx.sort(
            mx.where(valid_audio_mask, audio_indices, flat_audio.shape[0])
        )[:valid_audio_count]
        valid_audio = flat_audio[valid_audio_positions].astype(inputs_embeds.dtype)
        current = flat_inputs[placeholder_positions]
        flat_inputs = flat_inputs.at[placeholder_positions].add(valid_audio - current)
        return flat_inputs.reshape(inputs_embeds.shape)

    def prefill_speech(
        self,
        input_ids: mx.array,
        audio_input_embeds: mx.array,
        audio_input_embeds_mask: mx.array,
        *,
        cache: Optional[list[KVCache]] = None,
    ) -> tuple[mx.array, mx.array]:
        inputs_embeds = self.prepare_speech_embeddings(
            input_ids,
            audio_input_embeds,
            audio_input_embeds_mask,
        )
        return self.thinker(
            input_ids,
            cache=cache,
            input_embeddings=inputs_embeds,
        )

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        mapped = RaonTTSModel.sanitize(weights)
        audio_encoder = {
            name: value
            for name, value in weights.items()
            if name.startswith("audio_encoder.")
        }
        mapped.update(
            {
                f"speech.{name}": value
                for name, value in RaonSpeechComponents.sanitize(audio_encoder).items()
            }
        )
        input_adaptor_names = {
            "input_adaptor.proj.0.weight": "input_adaptor.linear_fc1.weight",
            "input_adaptor.proj.2.weight": "input_adaptor.linear_fc2.weight",
            "input_adaptor.post_norm.weight": "input_adaptor.post_norm.weight",
        }
        for source_name, target_name in input_adaptor_names.items():
            if source_name in weights:
                mapped[target_name] = weights[source_name]
        return mapped

    @staticmethod
    def _load_config(values: Dict[str, Any]) -> RaonSpeechConfig:
        return RaonSpeechConfig.from_dict(values)


class RaonDuplexModel(RaonSpeechModel):
    """One-frame composition boundary for source-compatible duplex decoding."""

    def __init__(self, config: RaonDuplexConfig, codec: Optional[Any] = None):
        super().__init__(config, codec=codec)
        self.config = config
        self.state_manager = DuplexStateManager(config.state)

    @staticmethod
    def _as_batch(input_ids: mx.array) -> mx.array:
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Raon duplex decoding currently requires batch size 1.")
        return input_ids

    def _select_text_token(
        self,
        text_logits: mx.array,
        machine_state: DuplexMachineState,
        sampler: Optional[Callable[[mx.array], Any]],
        forced_id: Optional[int] = None,
    ) -> int:
        if forced_id is not None:
            return forced_id
        if text_logits.ndim != 3 or text_logits.shape[1] < 2:
            raise ValueError(
                "Raon duplex text logits must include the token preceding audio output."
            )
        logits = text_logits[:, -2:-1, : self.config.thinker.vocab_size]
        if self.config.state.use_duplex_end_pad:
            logits = self.state_manager.apply_logit_mask(
                logits,
                machine_state,
                self.config.thinker.vocab_size,
            )
        selected = (
            mx.argmax(logits[:, -1], axis=-1)
            if sampler is None
            else sampler(logits[:, -1])
        )
        if isinstance(selected, mx.array):
            if selected.size != 1:
                raise ValueError("Raon duplex text sampler must return one token.")
            selected_id = int(selected.item())
        else:
            selected_id = int(selected)
        if not 0 <= selected_id < logits.shape[-1]:
            raise ValueError(
                f"Raon duplex text sampler selected out-of-range token {selected_id}."
            )
        if not bool(mx.isfinite(logits[0, -1, selected_id]).item()):
            raise ValueError(
                f"Raon duplex text sampler selected masked token {selected_id}."
            )
        return selected_id

    def _generate_duplex_codes(
        self,
        talker_hidden: mx.array,
        sampler: Optional[Callable[[mx.array], Any]],
    ) -> mx.array:
        def suppress_audio_end(logits: mx.array) -> Any:
            masked = logits.at[:, self.config.codebook_size].add(-mx.inf)
            return mx.argmax(masked, axis=-1) if sampler is None else sampler(masked)

        codes = self.speech.generate_audio_codes_from_talker_hidden(
            talker_hidden[:, -1:],
            first_code_sampler=suppress_audio_end,
        )
        ended = codes[:, 0] == self.config.codebook_size
        return mx.where(ended[:, None], mx.zeros_like(codes), codes)

    def _transition(
        self,
        *,
        state: RaonDuplexFrameState,
        talker_hidden: mx.array,
        text_logits: mx.array,
        text_sampler: Optional[Callable[[mx.array], Any]],
        first_code_sampler: Optional[Callable[[mx.array], Any]],
        forced_id: Optional[int] = None,
    ) -> RaonDuplexFrameResult:
        predicted_id = self._select_text_token(
            text_logits,
            state.machine_state,
            text_sampler,
            forced_id=forced_id,
        )
        machine_state, frame_tokens, emitted_audio = self.state_manager.transition(
            state.machine_state,
            predicted_id,
        )
        emitted_codes = (
            self._generate_duplex_codes(talker_hidden, first_code_sampler)
            if emitted_audio
            else None
        )
        audio_codes = state.audio_codes
        if emitted_codes is not None:
            audio_codes = mx.concatenate(
                [audio_codes, emitted_codes[:, None, :]], axis=1
            )
        frame = mx.array([frame_tokens], dtype=mx.int32)
        next_state = RaonDuplexFrameState(
            sequences=mx.concatenate([state.sequences, frame], axis=1),
            audio_codes=audio_codes,
            thinker_cache=state.thinker_cache,
            talker_cache=state.talker_cache,
            machine_state=machine_state,
            forced_sil_remaining=(
                state.forced_sil_remaining - 1
                if forced_id == self.config.state.duplex_sil_token_id
                and state.forced_sil_remaining
                else state.forced_sil_remaining
            ),
        )
        return RaonDuplexFrameResult(
            state=next_state,
            frame_tokens=frame_tokens,
            emitted_audio=emitted_audio,
            emitted_codes=emitted_codes,
        )

    def init_duplex_state(
        self,
        system_tokens: mx.array,
        *,
        speak_first: bool = False,
        text_sampler: Optional[Callable[[mx.array], Any]] = None,
        first_code_sampler: Optional[Callable[[mx.array], Any]] = None,
    ) -> RaonDuplexFrameState:
        system_tokens = self._as_batch(system_tokens)
        initial_tokens = mx.concatenate(
            [
                system_tokens,
                mx.array([[IM_START_ID, AUDIO_START_ID]], dtype=mx.int32),
            ],
            axis=1,
        )
        thinker_cache = self.thinker.make_cache()
        talker_cache = self.speech.talker.make_cache()
        talker_hidden, text_logits = self._forward(
            initial_tokens,
            thinker_cache,
            talker_cache,
        )
        empty_codes = mx.zeros(
            (1, 0, self.config.components.code_predictor.num_code_groups),
            dtype=mx.int32,
        )
        state = RaonDuplexFrameState(
            sequences=initial_tokens,
            audio_codes=empty_codes,
            thinker_cache=thinker_cache,
            talker_cache=talker_cache,
            machine_state=self.state_manager.initial_state(speak_first=speak_first),
            forced_sil_remaining=(
                2 if not speak_first and self.config.state.use_sil_token else 0
            ),
        )
        result = self._transition(
            state=state,
            talker_hidden=talker_hidden,
            text_logits=text_logits,
            text_sampler=text_sampler,
            first_code_sampler=first_code_sampler,
            forced_id=self.state_manager.initial_forced_prediction_id(speak_first),
        )
        return result.state

    def _prepare_duplex_embeddings(
        self,
        input_ids: mx.array,
        audio_input_embeds: mx.array,
        audio_input_embeds_mask: mx.array,
        previous_audio_codes: Optional[mx.array],
    ) -> mx.array:
        empty_streaming_frame = (
            audio_input_embeds.ndim == 3
            and audio_input_embeds.shape[1] == 0
            and audio_input_embeds_mask.ndim == 2
            and audio_input_embeds_mask.shape == audio_input_embeds.shape[:2]
        )
        if empty_streaming_frame:
            if input_ids.ndim != 2:
                raise ValueError(
                    "Raon speech input IDs must have shape (batch, sequence)."
                )
            if audio_input_embeds.shape[0] != input_ids.shape[0]:
                raise ValueError("Raon speech input batch dimensions must match.")
            if audio_input_embeds.shape[-1] != self.config.thinker.hidden_size:
                raise ValueError(
                    "Raon audio input embedding size must match the thinker hidden size."
                )
            input_count = int(mx.sum(input_ids == AUDIO_INPUT_PLACEHOLDER_ID).item())
            if input_count != 1:
                raise ValueError(
                    "Raon duplex empty streaming frames require exactly one "
                    "audio input placeholder."
                )
            inputs = self.thinker.embed_tokens(input_ids)
        else:
            inputs = self.prepare_speech_embeddings(
                input_ids,
                audio_input_embeds,
                audio_input_embeds_mask,
            )
        if previous_audio_codes is None:
            return inputs
        output_mask = input_ids == AUDIO_OUTPUT_PLACEHOLDER_ID
        output_count = int(mx.sum(output_mask).item())
        if output_count != 1:
            raise ValueError(
                "Raon duplex feedback requires exactly one audio output placeholder."
            )
        feedback = self.feedback_embedding(previous_audio_codes)[:, 0]
        positions = mx.where(
            output_mask.reshape(-1),
            mx.arange(input_ids.size),
            input_ids.size,
        )
        position = mx.sort(positions)[0]
        flat = inputs.reshape(-1, inputs.shape[-1])
        current = flat[position]
        flat = flat.at[position].add(feedback[0] - current)
        return flat.reshape(inputs.shape)

    def duplex_frame(
        self,
        state: RaonDuplexFrameState,
        *,
        audio_input_embeds: mx.array,
        audio_input_embeds_mask: mx.array,
        text_sampler: Optional[Callable[[mx.array], Any]] = None,
        first_code_sampler: Optional[Callable[[mx.array], Any]] = None,
    ) -> RaonDuplexFrameResult:
        frame_tokens = state.machine_state.last_frame_tokens
        input_ids = mx.array([frame_tokens], dtype=mx.int32)
        if state.sequences.shape[1] < len(frame_tokens) or not mx.array_equal(
            state.sequences[:, -len(frame_tokens) :], input_ids
        ):
            raise ValueError("Raon duplex state tokens do not match the sequence tail.")
        previous = state.audio_codes[:, -1:] if state.audio_codes.shape[1] else None
        inputs = self._prepare_duplex_embeddings(
            input_ids,
            audio_input_embeds,
            audio_input_embeds_mask,
            previous,
        )
        talker_hidden, text_logits = self._forward(
            input_ids,
            state.thinker_cache,
            state.talker_cache,
            input_embeddings=inputs,
        )
        return self._transition(
            state=state,
            talker_hidden=talker_hidden,
            text_logits=text_logits,
            text_sampler=text_sampler,
            first_code_sampler=first_code_sampler,
            forced_id=(
                self.config.state.duplex_sil_token_id
                if state.forced_sil_remaining
                else None
            ),
        )

    @staticmethod
    def _load_config(values: Dict[str, Any]) -> RaonDuplexConfig:
        return RaonDuplexConfig.from_dict(values)
