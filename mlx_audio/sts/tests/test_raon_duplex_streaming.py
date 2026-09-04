import unittest

import mlx.core as mx
import numpy as np

import mlx_audio.sts.models.raon as raon


class _StubAudioEncoder:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.frames = 0

    def step(self, audio_frame):
        self.frames += 1
        if self.frames == 1:
            return (
                mx.zeros((1, 0, self.hidden_size)),
                mx.zeros((1, 0), dtype=mx.bool_),
            )
        return (
            mx.full((1, 1, self.hidden_size), self.frames / 10),
            mx.ones((1, 1), dtype=mx.bool_),
        )


class _StubAudioDecoder:
    def __init__(self):
        self.codes = []

    def decode_frame(self, codes):
        self.codes.append(np.asarray(codes).copy())
        value = float(np.asarray(codes).sum())
        return mx.full((1, 1, 1920), value, dtype=mx.float32)


class _StubDuplexModel:
    sample_rate = 24_000

    def __init__(self):
        self.config = type(
            "Config",
            (),
            {
                "thinker": type("Thinker", (), {"hidden_size": 8})(),
                "components": type(
                    "Components",
                    (),
                    {
                        "code_predictor": type(
                            "CodePredictor", (), {"num_code_groups": 3}
                        )()
                    },
                )(),
            },
        )()
        self.frame_calls = []
        self.initial_state = object()
        self.tokenizer = None

    def init_duplex_state(self, system_tokens, **kwargs):
        self.system_tokens = np.asarray(system_tokens).copy()
        self.init_kwargs = kwargs
        return self.initial_state

    def duplex_frame(self, state, **kwargs):
        self.frame_calls.append(kwargs)
        emitted = len(self.frame_calls) == 2
        codes = mx.array([[1, 2, 3]], dtype=mx.int32) if emitted else None
        return type(
            "FrameResult",
            (),
            {
                "state": object(),
                "frame_tokens": [1, 2, 3],
                "emitted_audio": emitted,
                "emitted_codes": codes,
            },
        )()


class TestRaonDuplexStreaming(unittest.TestCase):
    def test_prepare_duplex_prompt_uses_system_message_without_generation_suffix(self):
        class Tokenizer:
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "rendered-system"

            def encode(self, text, add_special_tokens=False):
                self.encoded = (text, add_special_tokens)
                return [11, 12]

        tokenizer = Tokenizer()

        tokens = raon.prepare_duplex_prompt(tokenizer, "Be concise.")

        self.assertEqual(tokens.tolist(), [11, 12])
        self.assertEqual(
            tokenizer.messages,
            [{"role": "system", "content": "Be concise."}],
        )
        self.assertFalse(tokenizer.tokenize)
        self.assertFalse(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.encoded, ("rendered-system", False))

    def test_causal_mel_matches_source_startup_cadence(self):
        stream = raon.RaonCausalMelStream()

        first = stream.step(np.zeros(1280, dtype=np.float32))
        second = stream.step(np.zeros(1280, dtype=np.float32))

        self.assertEqual(first.shape, (128, 6))
        self.assertEqual(second.shape, (128, 8))
        self.assertTrue(np.isfinite(np.asarray(first)).all())
        self.assertTrue(np.isfinite(np.asarray(second)).all())

    def test_streaming_audio_encoder_withholds_first_frame(self):
        config = raon.RaonVoxtralEncoderConfig(
            dim=8,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            head_dim=4,
            hidden_dim=16,
            sliding_window=16,
            downsample_factor=4,
            skip_projector=True,
        )
        encoder = raon.RaonVoxtralEncoder(config)
        stream = raon.RaonStreamingAudioEncoder(encoder)
        frame = np.zeros(1920, dtype=np.float32)

        first, first_mask = stream.step(frame)
        second, second_mask = stream.step(frame)
        mx.eval(first, first_mask, second, second_mask)

        self.assertEqual(first.shape, (1, 0, 32))
        self.assertEqual(second.shape, (1, 1, 32))
        self.assertEqual(first_mask.shape, (1, 0))
        self.assertEqual(second_mask.tolist(), [[True]])
        self.assertTrue(np.isfinite(np.asarray(second)).all())

    def test_session_decodes_one_pcm_frame_per_user_frame(self):
        model = _StubDuplexModel()
        encoder = _StubAudioEncoder(hidden_size=8)
        decoder = _StubAudioDecoder()
        silence_codes = mx.array([[4, 5, 6]], dtype=mx.int32)
        session = raon.RaonDuplexSession(
            model,
            mx.array([7], dtype=mx.int32),
            audio_encoder=encoder,
            audio_decoder=decoder,
            silence_codes=silence_codes,
            speak_first=False,
        )

        first = session.step(np.zeros(1920, dtype=np.float32))
        second = session.step(np.ones(1920, dtype=np.float32))
        mx.eval(first.audio, second.audio)

        self.assertEqual(first.input_embedding_count, 0)
        self.assertEqual(second.input_embedding_count, 1)
        self.assertFalse(first.emitted_audio)
        self.assertTrue(second.emitted_audio)
        self.assertEqual(first.audio.shape, (1920,))
        self.assertEqual(second.audio.shape, (1920,))
        np.testing.assert_array_equal(decoder.codes[0], np.array([[4, 5, 6]]))
        np.testing.assert_array_equal(decoder.codes[1], np.array([[1, 2, 3]]))

    def test_fixed_input_yields_only_complete_source_frames(self):
        model = _StubDuplexModel()
        session = raon.RaonDuplexSession(
            model,
            mx.array([7], dtype=mx.int32),
            audio_encoder=_StubAudioEncoder(hidden_size=8),
            audio_decoder=_StubAudioDecoder(),
            silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
        )
        audio = np.zeros(1920 * 2 + 1111, dtype=np.float32)

        results = list(session.process(audio))

        self.assertEqual(len(results), 2)
        self.assertEqual([result.frame_index for result in results], [0, 1])

    def test_model_entrypoint_composes_fixed_input_session(self):
        model = _StubDuplexModel()
        audio = np.zeros(1920 * 2, dtype=np.float32)

        results = list(
            raon.RaonDuplexModel.generate_duplex(
                model,
                audio,
                system_tokens=mx.array([7], dtype=mx.int32),
                audio_encoder=_StubAudioEncoder(hidden_size=8),
                audio_decoder=_StubAudioDecoder(),
                silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(model.system_tokens.tolist(), [7])

    def test_model_entrypoint_uses_source_sampling_defaults(self):
        model = _StubDuplexModel()

        raon.RaonDuplexModel.create_duplex_session(
            model,
            system_tokens=mx.array([7], dtype=mx.int32),
            audio_encoder=_StubAudioEncoder(hidden_size=8),
            audio_decoder=_StubAudioDecoder(),
            silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
        )

        self.assertIsNotNone(model.init_kwargs["text_sampler"])
        self.assertIsNotNone(model.init_kwargs["first_code_sampler"])

    def test_model_entrypoint_defaults_to_empty_source_prompt(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "rendered-system"

            def encode(self, *args, **kwargs):
                return [11, 12]

        model = _StubDuplexModel()
        model.tokenizer = Tokenizer()

        raon.RaonDuplexModel.create_duplex_session(
            model,
            audio_encoder=_StubAudioEncoder(hidden_size=8),
            audio_decoder=_StubAudioDecoder(),
            silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
        )

        self.assertEqual(model.system_tokens.tolist(), [])

    def test_model_entrypoint_allows_explicit_greedy_decoding(self):
        model = _StubDuplexModel()

        raon.RaonDuplexModel.create_duplex_session(
            model,
            system_tokens=mx.array([7], dtype=mx.int32),
            do_sample=False,
            audio_encoder=_StubAudioEncoder(hidden_size=8),
            audio_decoder=_StubAudioDecoder(),
            silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
        )

        self.assertIsNone(model.init_kwargs["text_sampler"])
        self.assertIsNone(model.init_kwargs["first_code_sampler"])

    def test_model_entrypoint_rejects_invalid_sampling_controls(self):
        model = _StubDuplexModel()

        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            raon.RaonDuplexModel.create_duplex_session(
                model,
                system_tokens=mx.array([7], dtype=mx.int32),
                temperature=0,
                audio_encoder=_StubAudioEncoder(hidden_size=8),
                audio_decoder=_StubAudioDecoder(),
                silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
            )

    def test_session_rejects_wrong_frame_size(self):
        session = raon.RaonDuplexSession(
            _StubDuplexModel(),
            mx.array([7], dtype=mx.int32),
            audio_encoder=_StubAudioEncoder(hidden_size=8),
            audio_decoder=_StubAudioDecoder(),
            silence_codes=mx.array([[4, 5, 6]], dtype=mx.int32),
        )

        with self.assertRaisesRegex(ValueError, "exactly 1920"):
            session.step(np.zeros(1919, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
