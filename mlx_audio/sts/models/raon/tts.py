import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.codec.models.mimi.mimi import Mimi, mimi_202407
from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen3 import ModelArgs as Qwen3ModelArgs
from mlx_audio.lm.models.qwen3 import Qwen3Model
from mlx_audio.utils import get_model_path, load_weights

from .components import RaonComponentConfig, RaonSpeechComponents
from .duplex import AUDIO_OUTPUT_PLACEHOLDER_ID, AUDIO_START_ID

AUDIO_END_ID = 151670
IM_END_ID = 151645


@dataclass(frozen=True)
class RaonOutputAdaptorConfig:
    input_size: int
    output_size: int
    hidden_size: int
    num_layers: int
    output_time_scale: float
    use_post_norm: bool
    norm_eps: float

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonOutputAdaptorConfig":
        config = cls(
            input_size=int(values["input_size"]),
            output_size=int(values["output_size"]),
            hidden_size=int(values.get("hidden_size") or values["output_size"]),
            num_layers=int(values.get("num_layers", 1)),
            output_time_scale=float(values.get("output_time_scale", 1)),
            use_post_norm=bool(values.get("use_post_norm", False)),
            norm_eps=float(values.get("norm_eps", 1e-6)),
        )
        if config.num_layers != 2 or config.output_time_scale != 1:
            raise ValueError(
                "Raon TTS currently requires a two-layer output adaptor with "
                "output_time_scale=1."
            )
        return config


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


class RaonOutputAdaptor(nn.Module):
    def __init__(self, config: RaonOutputAdaptorConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.input_size, config.hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(config.hidden_size, config.output_size, bias=False)
        self.post_norm = (
            nn.RMSNorm(config.output_size, eps=config.norm_eps)
            if config.use_post_norm
            else None
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        outputs = self.linear_fc2(nn.gelu(self.linear_fc1(inputs)))
        if self.post_norm is not None:
            outputs = self.post_norm(outputs)
        return outputs


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

    def __init__(self, config: RaonTTSConfig, codec: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.thinker = RaonThinker(config.thinker)
        self.lm_head = nn.Linear(
            config.thinker.hidden_size, config.thinker.vocab_size, bias=False
        )
        self.output_adaptor = RaonOutputAdaptor(config.output_adaptor)
        self.speech = RaonSpeechComponents(config.components)
        self.speech.audio_encoder = None
        self.codec = codec or Mimi(
            mimi_202407(
                config.num_quantizers,
                transformers_compatible=True,
            )
        )
        self.tokenizer = None

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

    def generate(
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

    def generate_text(self, text: str, *, max_frames: int) -> RaonTTSResult:
        if self.tokenizer is None:
            raise RuntimeError("Raon tokenizer is not loaded.")
        return self.generate(
            prepare_tts_prompt(self.tokenizer, text),
            max_frames=max_frames,
        )

    def validate_source_families(self, weights: Dict[str, mx.array]) -> None:
        missing = [
            prefix
            for prefix in self._REQUIRED_SOURCE_FAMILIES
            if not any(name.startswith(prefix) for name in weights)
        ]
        if missing:
            raise ValueError(f"Raon TTS missing source families: {missing}.")

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
        codec = Mimi.sanitize_transformers_weights(weights, prefix="audio_tokenizer.")
        mapped.update({f"codec.{name}": value for name, value in codec.items()})
        return mapped

    def load_source_weights(self, weights: Dict[str, mx.array]) -> Dict[str, int]:
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
        return {"expected": len(expected), "admitted": len(received)}

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        *,
        revision: Optional[str] = None,
    ) -> "RaonTTSModel":
        model_path = get_model_path(path_or_repo, revision=revision)
        with open(Path(model_path) / "config.json", encoding="utf-8") as handle:
            config = cls._load_config(json.load(handle))
        model = cls(config)
        model.load_source_weights(load_weights(Path(model_path)))
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
