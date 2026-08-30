from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.stt.models.voxtral_realtime.config import EncoderConfig
from mlx_audio.stt.models.voxtral_realtime.encoder import AudioEncoder
from mlx_audio.tts.models.qwen3_tts.config import (
    Qwen3TTSTalkerCodePredictorConfig,
    Qwen3TTSTalkerConfig,
)
from mlx_audio.tts.models.qwen3_tts.talker import (
    CodePredictorDecoderLayer,
    RotaryEmbedding,
    TalkerDecoderLayer,
    TalkerRotaryEmbedding,
)


def _dataclass_kwargs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    valid = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in valid}


@dataclass
class RaonVoxtralEncoderConfig(EncoderConfig):
    skip_projector: bool = True
    traditional_rope: bool = False

    @property
    def stacked_output_size(self) -> int:
        return self.dim * self.downsample_factor

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonVoxtralEncoderConfig":
        aliases = {
            "hidden_size": "dim",
            "num_hidden_layers": "n_layers",
            "num_attention_heads": "n_heads",
            "num_key_value_heads": "n_kv_heads",
            "intermediate_size": "hidden_dim",
            "rms_norm_eps": "norm_eps",
        }
        normalized = {aliases.get(key, key): value for key, value in values.items()}
        return cls(**_dataclass_kwargs(cls, normalized))


@dataclass
class RaonComponentConfig:
    audio_encoder: RaonVoxtralEncoderConfig
    talker: Qwen3TTSTalkerConfig
    code_predictor: Qwen3TTSTalkerCodePredictorConfig
    thinker_hidden_size: int
    codebook_size: int
    projection_mode: str = "mlp"
    projection_intermediate_size: Optional[int] = None
    projection_pre_norm: bool = False
    proj_code_bias: bool = True

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "RaonComponentConfig":
        audio_encoder = RaonVoxtralEncoderConfig.from_dict(
            values["audio_encoder_config"]
        )
        if not audio_encoder.skip_projector:
            raise ValueError(
                "Raon component composition currently requires "
                "audio_encoder_config.skip_projector=true."
            )

        code_values = values["code_predictor_config"]
        code_predictor = Qwen3TTSTalkerCodePredictorConfig(
            **_dataclass_kwargs(Qwen3TTSTalkerCodePredictorConfig, code_values)
        )
        talker_values = values["talker_config"]
        talker = Qwen3TTSTalkerConfig(
            **_dataclass_kwargs(Qwen3TTSTalkerConfig, talker_values)
        )
        talker.code_predictor_config = code_predictor

        codebook_size = int(values["audio_tokenizer_config"]["codebook_size"])
        if codebook_size != code_predictor.vocab_size:
            raise ValueError(
                "Raon codebook size must match the code predictor vocabulary: "
                f"{codebook_size} != {code_predictor.vocab_size}."
            )

        return cls(
            audio_encoder=audio_encoder,
            talker=talker,
            code_predictor=code_predictor,
            thinker_hidden_size=int(values["text_model_config"]["hidden_size"]),
            codebook_size=codebook_size,
            projection_mode=values.get("thinker_to_talker_projection_mode", "linear"),
            projection_intermediate_size=values.get(
                "thinker_to_talker_intermediate_size"
            ),
            projection_pre_norm=values.get("thinker_to_talker_pre_norm", False),
            proj_code_bias=values.get("proj_code_bias", False),
        )


class ThinkerToTalkerProjection(nn.Module):
    def __init__(self, config: RaonComponentConfig):
        super().__init__()
        self.mode = config.projection_mode
        self.norm = (
            nn.RMSNorm(config.thinker_hidden_size, eps=config.talker.rms_norm_eps)
            if config.projection_pre_norm
            else None
        )
        if self.mode == "mlp":
            intermediate = config.projection_intermediate_size
            if intermediate is None:
                raise ValueError(
                    "thinker_to_talker_intermediate_size is required for MLP projection."
                )
            self.linear_fc1 = nn.Linear(
                config.thinker_hidden_size, intermediate, bias=True
            )
            self.linear_fc2 = nn.Linear(
                intermediate, config.talker.hidden_size, bias=True
            )
            self.linear = None
        elif self.mode == "linear":
            self.linear = nn.Linear(
                config.thinker_hidden_size, config.talker.hidden_size, bias=False
            )
            self.linear_fc1 = None
            self.linear_fc2 = None
        else:
            raise ValueError(f"Unsupported thinker-to-talker projection: {self.mode}")

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if self.mode == "mlp":
            return self.linear_fc2(nn.silu(self.linear_fc1(hidden_states)))
        return self.linear(hidden_states)


class RaonVoxtralEncoder(nn.Module):
    """Voxtral Realtime encoder ending at Raon's unprojected frame stack."""

    def __init__(self, config: RaonVoxtralEncoderConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.encoder.audio_language_projection_0 = None
        self.encoder.audio_language_projection_2 = None

    def __call__(self, mel: mx.array) -> mx.array:
        if mel.ndim == 2:
            mel = mel[None]
        if mel.ndim != 3:
            raise ValueError(
                "Raon audio features must have shape (batch, mel_bins, frames) "
                "or (mel_bins, frames)."
            )
        if mel.shape[0] == 0:
            raise ValueError("Raon audio feature batches cannot be empty.")
        return mx.stack([self._encode_one(sample) for sample in mel], axis=0)

    def _encode_one(self, mel: mx.array) -> mx.array:
        conv_out = self.encoder.conv_stem(mel)
        if conv_out.shape[0] <= self.config.sliding_window:
            hidden_states = conv_out
            for layer in self.encoder.transformer_layers:
                hidden_states = layer(hidden_states, 0, "causal")
            hidden_states = self.encoder.transformer_norm(hidden_states)
        else:
            hidden_states = mx.concatenate(
                list(self.encoder.encode_chunks(conv_out)), axis=0
            )

        usable = (
            hidden_states.shape[0] // self.config.downsample_factor
        ) * self.config.downsample_factor
        hidden_states = hidden_states[:usable]
        return hidden_states.reshape(-1, self.config.stacked_output_size)


class RaonTalker(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig):
        super().__init__()
        self.config = config
        self.layers = [
            TalkerDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = TalkerRotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            mrope_section=None,
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs_embeds.shape
        offset = cache[0].offset if cache and cache[0] is not None else 0
        if attention_mask is not None and offset:
            raise ValueError(
                "RaonTalker does not support cached explicit attention masks."
            )
        if position_ids is None:
            if attention_mask is not None:
                positions = (mx.cumsum(attention_mask, axis=-1) - 1).astype(mx.int32)
                positions = mx.maximum(positions, mx.zeros_like(positions))
                positions = positions[:, -seq_len:]
            else:
                positions = mx.broadcast_to(
                    mx.arange(offset, offset + seq_len)[None, :],
                    (batch, seq_len),
                )
            position_ids = mx.stack([positions, positions, positions], axis=0)
        elif position_ids.ndim == 2:
            position_ids = mx.stack([position_ids, position_ids, position_ids], axis=0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        if mask is None:
            if attention_mask is not None:
                pad_mask = (
                    1 - attention_mask[:, None, None, :].astype(inputs_embeds.dtype)
                ) * -1e9
                if seq_len > 1:
                    causal = nn.MultiHeadAttention.create_additive_causal_mask(
                        seq_len
                    ).astype(inputs_embeds.dtype)
                    mask = causal[None, None, :, :] + pad_mask
                else:
                    mask = pad_mask
            elif seq_len > 1:
                causal = create_attention_mask(
                    inputs_embeds,
                    cache[0] if cache else None,
                    return_array=True,
                )
                mask = mx.where(causal, 0.0, -mx.inf).astype(inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for index, layer in enumerate(self.layers):
            layer_cache = cache[index] if cache is not None else None
            hidden_states = layer(hidden_states, position_embeddings, mask, layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class RaonCodePredictorCore(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig):
        super().__init__()
        self.config = config
        self.layers = [
            CodePredictorDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[list[KVCache]] = None,
    ) -> mx.array:
        batch, seq_len, _ = inputs_embeds.shape
        offset = cache[0].offset if cache and cache[0] is not None else 0
        positions = mx.broadcast_to(
            mx.arange(offset, offset + seq_len)[None, :], (batch, seq_len)
        )
        position_embeddings = self.rotary_emb(inputs_embeds, positions)
        mask = None
        if seq_len > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
            mask = mask.astype(inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for index, layer in enumerate(self.layers):
            layer_cache = cache[index] if cache is not None else None
            hidden_states = layer(hidden_states, position_embeddings, mask, layer_cache)
        return self.norm(hidden_states)

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]


class RaonCodePredictor(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerCodePredictorConfig):
        super().__init__()
        self.config = config
        self.num_code_groups = config.num_code_groups
        self.codec_embedding = nn.Embedding(
            config.num_code_groups * config.vocab_size, config.hidden_size
        )
        self.fused_lm_head = mx.zeros(
            (
                config.num_code_groups - 1,
                config.vocab_size,
                config.hidden_size,
            )
        )
        self.model = RaonCodePredictorCore(config)

    def predict_codes(self, inputs_embeds: mx.array) -> mx.array:
        cache = self.model.make_cache()
        current = inputs_embeds
        generated = []
        for step in range(self.num_code_groups - 1):
            hidden_states = self.model(current, cache=cache)
            logits = hidden_states[:, -1] @ self.fused_lm_head[step].T
            token = mx.argmax(logits, axis=-1)
            generated.append(token)
            if step + 1 < self.num_code_groups - 1:
                token = token + (step + 1) * self.config.vocab_size
                current = self.codec_embedding(token)[:, None, :]
        return mx.stack(generated, axis=1)


class RaonSpeechComponents(nn.Module):
    _SOURCE_PREFIXES = (
        "audio_encoder.",
        "talker.",
        "code_predictor.",
        "thinker_to_talker_proj.",
        "audio_lm_head.",
        "proj_code.",
    )

    def __init__(self, config: RaonComponentConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = RaonVoxtralEncoder(config.audio_encoder)
        self.talker = RaonTalker(config.talker)
        self.thinker_to_talker_proj = ThinkerToTalkerProjection(config)
        self.audio_lm_head = nn.Linear(
            config.talker.hidden_size, config.codebook_size + 1, bias=False
        )
        self.proj_code = nn.Linear(
            config.talker.hidden_size,
            config.code_predictor.hidden_size,
            bias=config.proj_code_bias,
        )
        self.code_predictor = RaonCodePredictor(config.code_predictor)

    def encode_audio_features(self, mel: mx.array) -> mx.array:
        return self.audio_encoder(mel)

    def generate_audio_codes(
        self,
        thinker_hidden_states: mx.array,
        position_ids: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        talker_inputs = self.thinker_to_talker_proj(thinker_hidden_states)
        talker_hidden = self.talker(
            talker_inputs,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        return self.generate_audio_codes_from_talker_hidden(talker_hidden)

    def generate_audio_codes_from_talker_hidden(
        self,
        talker_hidden: mx.array,
        first_code_sampler: Optional[Callable[[mx.array], Any]] = None,
    ) -> mx.array:
        first_logits = self.audio_lm_head(talker_hidden[:, -1])
        if first_code_sampler is None:
            first_code = mx.argmax(first_logits, axis=-1)
        else:
            sampled = first_code_sampler(first_logits)
            first_code = (
                sampled
                if isinstance(sampled, mx.array)
                else mx.full((talker_hidden.shape[0],), int(sampled), mx.int32)
            )
            if first_code.ndim == 0:
                first_code = mx.broadcast_to(first_code, (talker_hidden.shape[0],))
            elif first_code.ndim == 2 and first_code.shape[-1] == 1:
                first_code = first_code[:, 0]
        ended = first_code == self.config.codebook_size
        safe_first = mx.minimum(first_code, self.config.codebook_size - 1)
        code_inputs = mx.concatenate(
            [
                self.proj_code(talker_hidden[:, -1:]),
                self.code_predictor.codec_embedding(safe_first)[:, None, :],
            ],
            axis=1,
        )
        remaining = self.code_predictor.predict_codes(code_inputs)
        remaining = mx.where(ended[:, None], mx.zeros_like(remaining), remaining)
        return mx.concatenate([first_code[:, None], remaining], axis=1)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        mapped: Dict[str, mx.array] = {}
        for source_name, value in weights.items():
            if not source_name.startswith(RaonSpeechComponents._SOURCE_PREFIXES):
                continue

            target_name = source_name
            if source_name.startswith("audio_encoder.encoder."):
                suffix = source_name.removeprefix("audio_encoder.encoder.")
                if suffix.startswith("embedder.conv1."):
                    suffix = suffix.replace(
                        "embedder.conv1.", "conv_layers_0_conv.conv.", 1
                    )
                elif suffix.startswith("embedder.conv2."):
                    suffix = suffix.replace(
                        "embedder.conv2.", "conv_layers_1_conv.conv.", 1
                    )
                elif suffix.startswith("layers."):
                    suffix = suffix.replace("layers.", "transformer_layers.", 1)
                    suffix = suffix.replace("self_attn.q_proj.", "attention.wq.")
                    suffix = suffix.replace("self_attn.k_proj.", "attention.wk.")
                    suffix = suffix.replace("self_attn.v_proj.", "attention.wv.")
                    suffix = suffix.replace("self_attn.o_proj.", "attention.wo.")
                    suffix = suffix.replace("self_attn_layer_norm.", "attention_norm.")
                    suffix = suffix.replace("final_layer_norm.", "ffn_norm.")
                    suffix = suffix.replace("mlp.gate_proj.", "feed_forward_w1.")
                    suffix = suffix.replace("mlp.up_proj.", "feed_forward_w3.")
                    suffix = suffix.replace("mlp.down_proj.", "feed_forward_w2.")
                elif suffix.startswith("norm."):
                    suffix = suffix.replace("norm.", "transformer_norm.", 1)
                target_name = f"audio_encoder.encoder.{suffix}"
                if source_name.endswith(
                    ("embedder.conv1.weight", "embedder.conv2.weight")
                ):
                    value = value.transpose(0, 2, 1)

            mapped[target_name] = value
        return mapped

    def load_component_weights(self, weights: Dict[str, mx.array]) -> Dict[str, int]:
        mapped = self.sanitize(weights)
        expected = {name for name, _ in tree_flatten(self.parameters())}
        received = set(mapped)
        missing = sorted(expected - received)
        unexpected = sorted(received - expected)
        if missing or unexpected:
            raise ValueError(
                "Raon component weight admission failed: "
                f"missing={missing}, unexpected={unexpected}."
            )
        self.load_weights(list(mapped.items()), strict=True)
        return {"expected": len(expected), "admitted": len(received)}
