import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

import mlx_audio.sts.models.raon as raon
from mlx_audio.sts.models.raon import (
    AUDIO_INPUT_PLACEHOLDER_ID,
    AUDIO_OUTPUT_BACKCHANNEL_ID,
    AUDIO_OUTPUT_END_PAD_ID,
    AUDIO_OUTPUT_PAD_ID,
    AUDIO_OUTPUT_PLACEHOLDER_ID,
    AUDIO_OUTPUT_SIL_ID,
    AUDIO_START_ID,
    DuplexMachineState,
    DuplexPhase,
    DuplexStateConfig,
    DuplexStateManager,
)


class TestRaonDuplexStateManager(unittest.TestCase):
    def setUp(self):
        self.manager = DuplexStateManager(
            DuplexStateConfig(
                use_duplex_end_pad=True,
                use_sil_token=True,
                sequence_mode="uta",
                use_backchannel_token=True,
            )
        )

    def test_backchannel_onset_and_return_to_silence(self):
        state = self.manager.initial_state()
        self.assertEqual(state.phase, DuplexPhase.SIL)
        self.assertEqual(
            state.last_frame_tokens,
            [AUDIO_INPUT_PLACEHOLDER_ID, AUDIO_OUTPUT_PLACEHOLDER_ID],
        )
        self.assertEqual(
            self.manager.initial_forced_prediction_id(speak_first=False),
            AUDIO_OUTPUT_SIL_ID,
        )
        self.assertEqual(
            self.manager.initial_forced_prediction_id(speak_first=True),
            AUDIO_OUTPUT_END_PAD_ID,
        )

        state, frame, emitted_audio = self.manager.transition(
            state, AUDIO_OUTPUT_BACKCHANNEL_ID
        )
        self.assertEqual(state.phase, DuplexPhase.SPEECH)
        self.assertEqual(
            frame,
            [
                AUDIO_INPUT_PLACEHOLDER_ID,
                AUDIO_OUTPUT_BACKCHANNEL_ID,
                AUDIO_OUTPUT_PLACEHOLDER_ID,
            ],
        )
        self.assertTrue(emitted_audio)

        state, frame, emitted_audio = self.manager.transition(
            state, AUDIO_OUTPUT_SIL_ID
        )
        self.assertEqual(state.phase, DuplexPhase.SIL)
        self.assertEqual(
            frame, [AUDIO_INPUT_PLACEHOLDER_ID, AUDIO_OUTPUT_PLACEHOLDER_ID]
        )
        self.assertTrue(emitted_audio)

    def test_silence_mask_only_allows_checkpoint_onset_tokens(self):
        logits = mx.zeros((1, 1, AUDIO_OUTPUT_END_PAD_ID + 1))
        masked = self.manager.apply_logit_mask(
            logits, self.manager.initial_state(), vocab_size=151679
        )

        allowed = np.flatnonzero(np.isfinite(np.array(masked[0, 0]))).tolist()
        self.assertEqual(
            allowed,
            [
                AUDIO_OUTPUT_SIL_ID,
                AUDIO_OUTPUT_BACKCHANNEL_ID,
                AUDIO_OUTPUT_END_PAD_ID,
            ],
        )
        self.assertFalse(mx.isfinite(masked[0, 0, AUDIO_OUTPUT_PAD_ID]).item())

    def test_audio_start_frame_counts_as_emitted_audio(self):
        state = DuplexMachineState(DuplexPhase.SPEECH, [AUDIO_START_ID])

        self.assertTrue(state.emitted_audio)

    def test_state_config_rejects_unsupported_sequence_modes(self):
        with self.assertRaisesRegex(ValueError, "sequence_mode"):
            DuplexStateConfig(sequence_mode="not-a-raon-mode")

    def test_state_config_rejects_unimplemented_no_audio_in_sil(self):
        with self.assertRaisesRegex(ValueError, "no_audio_in_sil"):
            DuplexStateConfig(no_audio_in_sil=True)


class TestRaonSpeechComponents(unittest.TestCase):
    def _component_types(self):
        names = (
            "RaonComponentConfig",
            "RaonSpeechComponents",
            "RaonVoxtralEncoderConfig",
        )
        missing = [name for name in names if not hasattr(raon, name)]
        self.assertEqual(
            missing,
            [],
            f"Raon component composition API is missing: {missing}",
        )
        return tuple(getattr(raon, name) for name in names)

    def _tiny_config(self):
        component_config, _, _ = self._component_types()
        return component_config.from_dict(
            {
                "text_model_config": {"hidden_size": 8},
                "audio_encoder_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "rms_norm_eps": 1e-5,
                    "rope_theta": 1_000_000.0,
                    "sliding_window": 16,
                    "downsample_factor": 2,
                    "skip_projector": True,
                },
                "talker_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10_000.0,
                    "vocab_size": 32,
                },
                "code_predictor_config": {
                    "vocab_size": 5,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1_000_000.0,
                    "num_code_groups": 3,
                },
                "audio_tokenizer_config": {"codebook_size": 5},
                "thinker_to_talker_projection_mode": "mlp",
                "thinker_to_talker_intermediate_size": 16,
                "thinker_to_talker_pre_norm": False,
                "proj_code_bias": True,
            }
        )

    def test_checkpoint_config_preserves_raon_component_boundaries(self):
        config = self._tiny_config()

        self.assertTrue(config.audio_encoder.skip_projector)
        self.assertEqual(config.audio_encoder.stacked_output_size, 16)
        self.assertEqual(config.talker.rope_theta, 10_000.0)
        self.assertEqual(config.code_predictor.num_code_groups, 3)

    def test_voxtral_boundary_emits_unprojected_frame_stacks(self):
        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        mel = mx.arange(128 * 12, dtype=mx.float32).reshape(1, 128, 12) / 1000

        encoded = model.encode_audio_features(mel)
        mx.eval(encoded)

        self.assertEqual(encoded.ndim, 3)
        self.assertEqual(encoded.shape[0], 1)
        self.assertEqual(encoded.shape[-1], 16)

    def test_voxtral_boundary_preserves_batch_and_example_isolation(self):
        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        first = mx.arange(128 * 12, dtype=mx.float32).reshape(128, 12) / 1000
        second = first * 0.5 + 0.125
        batch = mx.stack([first, second])

        together = model.encode_audio_features(batch)
        isolated = mx.concatenate(
            [
                model.encode_audio_features(first[None]),
                model.encode_audio_features(second[None]),
            ],
            axis=0,
        )
        mx.eval(together, isolated)

        self.assertEqual(together.shape, isolated.shape)
        self.assertEqual(together.shape[0], 2)
        np.testing.assert_allclose(
            np.array(together), np.array(isolated), rtol=1e-5, atol=1e-5
        )

    def test_voxtral_attention_uses_pinned_split_half_rope(self):
        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        attention = model.audio_encoder.encoder.transformer_layers[0].attention
        apply_rope = getattr(attention, "apply_rope", None)
        self.assertTrue(
            callable(apply_rope),
            "The effective Voxtral attention route must expose its rotary operation.",
        )
        values = mx.arange(1 * 2 * 3 * 4, dtype=mx.float32).reshape(1, 2, 3, 4)

        actual = apply_rope(values, offset=5)
        positions = mx.arange(5, 8, dtype=mx.float32)
        inv_freq = 1.0 / (1_000_000.0 ** (mx.arange(0, 4, 2, dtype=mx.float32) / 4))
        frequencies = positions[:, None] * inv_freq[None, :]
        embeddings = mx.concatenate([frequencies, frequencies], axis=-1)
        cos = mx.cos(embeddings)[None, None]
        sin = mx.sin(embeddings)[None, None]
        rotated = mx.concatenate([-values[..., 2:], values[..., :2]], axis=-1)
        expected = values * cos + rotated * sin
        mx.eval(actual, expected)

        np.testing.assert_allclose(
            np.array(actual), np.array(expected), rtol=1e-5, atol=1e-5
        )

    def test_talker_and_code_predictor_transition_is_deterministic(self):
        _, components_type, _ = self._component_types()
        mx.random.seed(17)
        model = components_type(self._tiny_config())
        thinker_hidden = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8) / 16
        position_ids = mx.array([[0, 1]])

        first = model.generate_audio_codes(thinker_hidden, position_ids=position_ids)
        second = model.generate_audio_codes(thinker_hidden, position_ids=position_ids)
        mx.eval(first, second)

        self.assertEqual(first.shape, (1, 3))
        np.testing.assert_array_equal(np.array(first), np.array(second))

    def test_code_predictor_preserves_checkpoint_hidden_cache_and_logits(self):
        class RecordingCore(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.input_dtypes = []
                self.output_dtypes = []
                self.cache_dtypes = []

            def make_cache(self):
                return self.inner.make_cache()

            def __call__(self, inputs_embeds, cache=None):
                self.input_dtypes.append(inputs_embeds.dtype)
                output = self.inner(inputs_embeds, cache=cache)
                mx.eval(output)
                self.output_dtypes.append(output.dtype)
                self.cache_dtypes.append(
                    [(layer.keys.dtype, layer.values.dtype) for layer in cache]
                )
                return output

        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        model.code_predictor.set_dtype(mx.bfloat16)
        recording = RecordingCore(model.code_predictor.model)
        model.code_predictor.model = recording
        inputs = mx.arange(16, dtype=mx.bfloat16).reshape(1, 2, 8) / 16
        logits_dtypes = []
        original_argmax = mx.argmax

        def recording_argmax(logits, *args, **kwargs):
            logits_dtypes.append(logits.dtype)
            return original_argmax(logits, *args, **kwargs)

        with mock.patch(
            "mlx_audio.sts.models.raon.components.mx.argmax",
            side_effect=recording_argmax,
        ):
            codes = model.code_predictor.predict_codes(inputs)
        mx.eval(codes)

        self.assertEqual(recording.input_dtypes, [mx.bfloat16, mx.bfloat16])
        self.assertEqual(recording.output_dtypes, [mx.bfloat16, mx.bfloat16])
        self.assertEqual(
            recording.cache_dtypes,
            [[(mx.bfloat16, mx.bfloat16)], [(mx.bfloat16, mx.bfloat16)]],
        )
        self.assertEqual(logits_dtypes, [mx.bfloat16, mx.bfloat16])

    def test_code_predictor_attention_uses_source_eager_path(self):
        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        model.code_predictor.set_dtype(mx.bfloat16)
        core = model.code_predictor.model
        attention = core.layers[0].self_attn
        inputs = mx.arange(16, dtype=mx.bfloat16).reshape(1, 2, 8) / 16
        positions = mx.array([[0, 1]])
        position_embeddings = core.rotary_emb(inputs, positions)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(2).astype(inputs.dtype)

        with (
            mock.patch(
                "mlx_audio.tts.models.qwen3_tts.talker.mx.fast.scaled_dot_product_attention",
                side_effect=AssertionError(
                    "fused SDPA must not serve source-eager code predictor attention"
                ),
            ),
            mock.patch(
                "mlx_audio.tts.models.qwen3_tts.talker.mx.softmax",
                wraps=mx.softmax,
            ) as softmax,
        ):
            output = attention(
                inputs,
                position_embeddings,
                mask=mask,
                cache=core.make_cache()[0],
            )
        mx.eval(output)

        self.assertEqual(output.shape, inputs.shape)
        self.assertEqual(output.dtype, mx.bfloat16)
        self.assertEqual(softmax.call_args.args[0].dtype, mx.float32)

    def test_component_loading_preserves_checkpoint_dtype_through_prediction(self):
        _, components_type, _ = self._component_types()

        for checkpoint_dtype in (mx.bfloat16, mx.float16):
            with self.subTest(checkpoint_dtype=checkpoint_dtype):
                source = components_type(self._tiny_config())
                checkpoint_weights = {
                    name: mx.zeros(value.shape, dtype=checkpoint_dtype)
                    for name, value in tree_flatten(source.parameters())
                }
                model = components_type(self._tiny_config())
                receipt = model.load_component_weights(checkpoint_weights)
                before = [
                    (value.dtype, value.nbytes)
                    for _, value in tree_flatten(model.parameters())
                ]

                codes = model.generate_audio_codes(
                    mx.zeros((1, 2, model.config.thinker_hidden_size))
                )
                mx.eval(codes)
                after = [
                    (value.dtype, value.nbytes)
                    for _, value in tree_flatten(model.parameters())
                ]

                self.assertEqual(receipt["expected"], receipt["admitted"])
                self.assertEqual(codes.shape, (1, 3))
                self.assertTrue(before)
                self.assertEqual(before, after)
                self.assertEqual({dtype for dtype, _ in after}, {checkpoint_dtype})

    def test_talker_rejects_explicit_attention_mask_with_nonempty_cache(self):
        _, components_type, _ = self._component_types()
        model = components_type(self._tiny_config())
        cache = model.talker.make_cache()
        inputs = mx.zeros((1, 1, model.config.talker.hidden_size))
        model.talker(inputs, cache=cache)

        with self.assertRaisesRegex(ValueError, "cached explicit attention masks"):
            model.talker(
                inputs,
                attention_mask=mx.ones((1, 2)),
                cache=cache,
            )

    def test_sanitizer_maps_pinned_raon_source_names(self):
        _, components_type, _ = self._component_types()
        weights = {
            "audio_encoder.encoder.embedder.conv1.weight": mx.zeros((8, 128, 3)),
            "audio_encoder.encoder.layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
            "audio_encoder.encoder.layers.0.final_layer_norm.weight": mx.zeros((8,)),
            "talker.layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
            "code_predictor.codec_embedding.weight": mx.zeros((15, 8)),
            "code_predictor.fused_lm_head": mx.zeros((2, 5, 8)),
        }

        mapped = components_type.sanitize(weights)

        self.assertEqual(
            mapped["audio_encoder.encoder.conv_layers_0_conv.conv.weight"].shape,
            (8, 3, 128),
        )
        self.assertIn(
            "audio_encoder.encoder.transformer_layers.0.attention.wq.weight",
            mapped,
        )
        self.assertIn(
            "audio_encoder.encoder.transformer_layers.0.ffn_norm.weight",
            mapped,
        )
        self.assertIn("talker.layers.0.self_attn.q_proj.weight", mapped)
        self.assertIn("code_predictor.codec_embedding.weight", mapped)
        self.assertIn("code_predictor.fused_lm_head", mapped)


if __name__ == "__main__":
    unittest.main()
