import unittest

import mlx.core as mx
import numpy as np

import mlx_audio.sts.models.raon as raon
from mlx_audio.sts.voice_pipeline import PocketTTSResponder
from mlx_audio.tts.models.base import GenerationResult


class _FakeQuantizer:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def decode(self, codes: mx.array) -> mx.array:
        values = mx.sum(codes.astype(mx.float32), axis=1, keepdims=True)
        scales = mx.arange(1, self.dimension + 1, dtype=mx.float32)[None, :, None]
        return values * scales / self.dimension


class _FakeCodec:
    sample_rate = 24_000
    frame_rate = 12.5

    def __init__(self, dimension: int):
        self.quantizer = _FakeQuantizer(dimension)

    def decode(self, codes: mx.array) -> mx.array:
        frame_values = mx.sum(codes.astype(mx.float32), axis=1)
        return mx.repeat(frame_values[:, None, :], 1920, axis=-1)


class _FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<|im_end|>":
            return [151645]
        assert text == "rendered"
        return [7, 151670, 151645]


class TestRaonTextToSpeech(unittest.TestCase):
    def _types(self):
        names = ("RaonTTSConfig", "RaonTTSModel", "prepare_tts_prompt")
        missing = [name for name in names if not hasattr(raon, name)]
        self.assertEqual(missing, [], f"Raon TTS integration API is missing: {missing}")
        return tuple(getattr(raon, name) for name in names)

    def _config(self, config_type=None):
        if config_type is None:
            config_type, _, _ = self._types()
        return config_type.from_dict(
            {
                "text_model_config": {
                    "model_type": "qwen3",
                    "hidden_size": 8,
                    "num_hidden_layers": 2,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 10_000.0,
                    "vocab_size": 151680,
                    "tie_word_embeddings": False,
                },
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
                "audio_tokenizer_config": {
                    "codebook_size": 5,
                    "hidden_size": 8,
                    "num_quantizers": 3,
                },
                "output_adaptor_config": {
                    "input_size": 8,
                    "output_size": 8,
                    "hidden_size": 16,
                    "num_layers": 2,
                    "output_time_scale": 1,
                    "use_post_norm": True,
                    "norm_eps": 1e-6,
                },
                "input_adaptor_config": {
                    "input_size": 16,
                    "output_size": 8,
                    "hidden_size": 8,
                    "num_layers": 2,
                    "output_time_scale": 1,
                    "use_post_norm": True,
                    "post_norm_init_scale": 0.02,
                    "norm_eps": 1e-6,
                },
                "thinker_to_talker_projection_mode": "mlp",
                "thinker_to_talker_intermediate_size": 16,
                "thinker_to_talker_pre_norm": False,
                "proj_code_bias": True,
            }
        )

    def test_prompt_uses_source_template_and_forces_audio_start(self):
        _, _, prepare_prompt = self._types()
        tokenizer = _FakeTokenizer()

        input_ids = prepare_prompt(tokenizer, "A compact source-authentic prompt.")

        self.assertEqual(input_ids.tolist(), [7, raon.AUDIO_START_ID])
        self.assertEqual(
            tokenizer.messages,
            [
                {
                    "role": "user",
                    "content": "Speak the following text:\nA compact source-authentic prompt.",
                }
            ],
        )

    def test_generated_codes_feed_back_and_both_caches_advance(self):
        _, model_type, _ = self._types()
        mx.random.seed(19)
        model = model_type(self._config(), codec=_FakeCodec(8))
        sampled = iter([1, 2, 3])

        result = model.generate_frames(
            mx.array([7, raon.AUDIO_START_ID]),
            max_frames=3,
            first_code_sampler=lambda _: next(sampled),
        )
        mx.eval(result.audio_codes, result.audio)

        self.assertEqual(result.finish_reason, "length")
        self.assertEqual(result.audio_codes.shape, (1, 3, 3))
        self.assertEqual(
            [step.thinker_cache_offset for step in result.steps], [2, 3, 4]
        )
        self.assertEqual([step.talker_cache_offset for step in result.steps], [2, 3, 4])
        self.assertEqual(
            [step.used_previous_audio for step in result.steps],
            [False, True, True],
        )

        first = model.feedback_embedding(mx.array([[[1, 0, 0]]]))
        second = model.feedback_embedding(mx.array([[[2, 0, 0]]]))
        mx.eval(first, second)
        self.assertFalse(np.allclose(np.array(first), np.array(second)))

    def test_audio_end_is_terminal_and_not_decoded_as_a_frame(self):
        _, model_type, _ = self._types()
        model = model_type(self._config(), codec=_FakeCodec(8))
        sampled = iter([2, self._config().codebook_size])

        result = model.generate_frames(
            mx.array([7, raon.AUDIO_START_ID]),
            max_frames=5,
            first_code_sampler=lambda _: next(sampled),
        )
        mx.eval(result.audio_codes, result.audio)

        self.assertEqual(result.finish_reason, "audio_end")
        self.assertEqual(result.audio_codes.shape[1], 1)
        self.assertEqual(result.steps[-1].first_code, self._config().codebook_size)
        self.assertEqual(result.audio.shape, (1, 1, 1920))

    def test_public_generation_composes_through_responder_and_trims_final_frame(self):
        _, model_type, _ = self._types()
        model = model_type(self._config(), codec=_FakeCodec(8))
        model.tokenizer = _FakeTokenizer()
        model.max_frames = 3
        model.speech.audio_lm_head.weight = mx.zeros_like(
            model.speech.audio_lm_head.weight
        )

        chunks = list(PocketTTSResponder(model).create_generator("hello"))

        self.assertEqual(len(chunks), 1)
        self.assertIsInstance(chunks[0], GenerationResult)
        self.assertEqual(chunks[0].audio.shape, (3840,))
        self.assertEqual(chunks[0].samples, 3840)
        self.assertEqual(chunks[0].sample_rate, 24_000)
        self.assertEqual(chunks[0].segment_idx, 0)
        self.assertEqual(chunks[0].token_count, 3)
        self.assertTrue(chunks[0].is_final_chunk)
        self.assertFalse(chunks[0].is_streaming_chunk)
        self.assertEqual(chunks[0].prompt["text"], "hello")

    def test_source_weight_admission_fails_loud_on_missing_family(self):
        _, model_type, _ = self._types()
        model = model_type(self._config(), codec=_FakeCodec(8))

        with self.assertRaisesRegex(ValueError, "missing source families"):
            model.validate_source_families({"lm_head.weight": mx.zeros((151680, 8))})

    def test_source_classification_rejects_unknown_root_and_supported_family(self):
        _, model_type, _ = self._types()
        model = model_type(self._config(), codec=_FakeCodec(8))

        for name in ("future_module.weight", "lm_head.unrecognized"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "unclassified source tensor"):
                    model.classify_source_weights({name: mx.zeros((1,))})

    def test_source_classification_records_explicit_tts_exclusions(self):
        _, model_type, _ = self._types()
        model = model_type(self._config(), codec=_FakeCodec(8))
        excluded = {
            "audio_encoder.layer.weight": mx.zeros((1,)),
            "input_adaptor.proj.weight": mx.zeros((1,)),
            "speaker_encoder.layer.weight": mx.zeros((1,)),
        }

        receipt = model.classify_source_weights(excluded)

        self.assertEqual(receipt["source_tensor_count"], 3)
        self.assertEqual(receipt["excluded_source_count"], 3)
        self.assertEqual(
            receipt["excluded_families"],
            {
                "audio_encoder.": 1,
                "input_adaptor.": 1,
                "speaker_encoder.": 1,
            },
        )
        self.assertEqual(receipt["unclassified_source_count"], 0)

    def test_speech_conditioning_adapts_voxtral_frames_and_preserves_mask(self):
        names = ("RaonSpeechConfig", "RaonSpeechModel")
        missing = [name for name in names if not hasattr(raon, name)]
        self.assertEqual(
            missing,
            [],
            f"Raon speech-conditioning API is missing: {missing}",
        )
        config_type, model_type = (getattr(raon, name) for name in names)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        encoded = mx.arange(2 * 3 * 16, dtype=mx.float32).reshape(2, 3, 16) / 100
        mask = mx.array([[True, True, False], [True, False, False]])

        adapted, output_mask = model.adapt_audio_embeddings(encoded, mask)
        mx.eval(adapted, output_mask)

        self.assertEqual(adapted.shape, (2, 3, 8))
        np.testing.assert_array_equal(np.array(output_mask), np.array(mask))
        self.assertIsNotNone(model.speech.audio_encoder)

    def test_speech_conditioning_composes_voxtral_through_input_adaptor(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        mel = mx.arange(128 * 12, dtype=mx.float32).reshape(1, 128, 12) / 1000

        adapted, mask = model.get_audio_input_embeds(mel)
        mx.eval(adapted, mask)

        self.assertEqual(adapted.ndim, 3)
        self.assertEqual(adapted.shape[0], 1)
        self.assertEqual(adapted.shape[-1], 8)
        self.assertEqual(mask.shape, adapted.shape[:2])
        self.assertTrue(np.array(mask).all())

    def test_speech_conditioning_inserts_only_valid_audio_frames(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        model.thinker.embed_tokens.weight = mx.zeros_like(
            model.thinker.embed_tokens.weight
        )
        first = mx.arange(1, 9, dtype=mx.float32)
        second = mx.arange(11, 19, dtype=mx.float32)
        ignored = mx.arange(21, 29, dtype=mx.float32)
        audio = mx.stack([first, second, ignored])[None]
        input_ids = mx.array(
            [[7, raon.AUDIO_INPUT_PLACEHOLDER_ID, 8, raon.AUDIO_INPUT_PLACEHOLDER_ID]]
        )

        embeddings = model.prepare_speech_embeddings(
            input_ids,
            audio,
            mx.array([[True, True, False]]),
        )
        mx.eval(embeddings)

        expected = np.zeros((1, 4, 8), dtype=np.float32)
        expected[0, 1] = np.arange(1, 9, dtype=np.float32)
        expected[0, 3] = np.arange(11, 19, dtype=np.float32)
        np.testing.assert_array_equal(np.array(embeddings), expected)

    def test_speech_conditioning_rejects_placeholder_count_mismatch(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        input_ids = mx.array([[raon.AUDIO_INPUT_PLACEHOLDER_ID, 7]])
        audio = mx.zeros((1, 2, 8))

        with self.assertRaisesRegex(ValueError, "placeholder count"):
            model.prepare_speech_embeddings(
                input_ids,
                audio,
                mx.array([[True, True]]),
            )

    def test_speech_conditioning_rejects_compensating_batch_mismatches(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        input_ids = mx.array(
            [
                [raon.AUDIO_INPUT_PLACEHOLDER_ID, 7],
                [raon.AUDIO_INPUT_PLACEHOLDER_ID, 8],
            ]
        )
        audio = mx.zeros((2, 2, 8))

        with self.assertRaisesRegex(ValueError, "batch row 0"):
            model.prepare_speech_embeddings(
                input_ids,
                audio,
                mx.array([[True, True], [False, False]]),
            )

    def test_speech_conditioned_prefill_advances_cache_across_inserted_audio(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        mx.random.seed(29)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))
        input_ids = mx.array([[7, raon.AUDIO_INPUT_PLACEHOLDER_ID, 8]])
        audio = mx.arange(8, dtype=mx.float32).reshape(1, 1, 8) / 10
        cache = model.thinker.make_cache()

        accepted, normalized = model.prefill_speech(
            input_ids,
            audio,
            mx.array([[True]]),
            cache=cache,
        )
        mx.eval(accepted, normalized)

        self.assertEqual(accepted.shape, (1, 3, 8))
        self.assertEqual(normalized.shape, (1, 3, 8))
        self.assertEqual([layer.offset for layer in cache], [3, 3])
        self.assertTrue(np.isfinite(np.array(normalized)).all())

    def test_speech_source_classification_rejects_unknown_input_adaptor_member(self):
        config_type = getattr(raon, "RaonSpeechConfig", None)
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(config_type)
        self.assertIsNotNone(model_type)
        model = model_type(self._config(config_type), codec=_FakeCodec(8))

        with self.assertRaisesRegex(ValueError, "inside supported families"):
            model.classify_source_weights(
                {"input_adaptor.unrecognized": mx.zeros((1,))}
            )

    def test_speech_source_sanitize_maps_all_input_adaptor_members(self):
        model_type = getattr(raon, "RaonSpeechModel", None)
        self.assertIsNotNone(model_type)
        weights = {
            "input_adaptor.proj.0.weight": mx.zeros((8, 16)),
            "input_adaptor.proj.2.weight": mx.zeros((8, 8)),
            "input_adaptor.post_norm.weight": mx.zeros((8,)),
        }

        mapped = model_type.sanitize(weights)

        self.assertEqual(
            set(mapped),
            {
                "input_adaptor.linear_fc1.weight",
                "input_adaptor.linear_fc2.weight",
                "input_adaptor.post_norm.weight",
            },
        )


if __name__ == "__main__":
    unittest.main()
