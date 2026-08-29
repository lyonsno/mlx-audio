import unittest
from unittest.mock import patch

import mlx.core as mx

from ..models.mimi.mimi import Mimi, mimi_202407
from ..models.mimi.modules.conv import ConvTranspose1d
from ..models.mimi.modules.quantization import EuclideanCodebook
from ..models.mimi.modules.transformer import MlpNoGating


class TestMimi(unittest.TestCase):
    def test_mimi_model(self):
        """Test Mimi model encoding and decoding."""
        model = Mimi(mimi_202407(32))

        audio = mx.zeros((1, 1, 120_000))
        codes = model.encode(audio)
        self.assertEqual(codes.shape, (1, 32, 63))

        audio_out = model.decode(codes)
        self.assertEqual(audio_out.shape, (1, 1, 120_960))

    def test_convtranspose_materializes_expanded_weight(self):
        with patch("mlx_audio.codec.models.mimi.modules.conv.mx.eval") as eval_mock:
            layer = ConvTranspose1d(4, 4, 3, groups=4)

        args = eval_mock.call_args.args
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], layer._expanded_weight)

    def test_codebook_materializes_derived_lookup_arrays(self):
        with patch(
            "mlx_audio.codec.models.mimi.modules.quantization.mx.eval"
        ) as eval_mock:
            codebook = EuclideanCodebook(dim=4, codebook_size=8)

        args = eval_mock.call_args.args
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], codebook._embedding)
        self.assertIs(args[1], codebook._c2)

    def test_from_pretrained_materializes_loaded_parameters(self):
        with (
            patch(
                "mlx_audio.codec.models.mimi.mimi.hf_hub_download",
                return_value="weights.safetensors",
            ),
            patch.object(Mimi, "load_pytorch_weights", return_value=None),
            patch("mlx_audio.codec.models.mimi.mimi.mx.eval") as eval_mock,
        ):
            Mimi.from_pretrained("test/repo")

        self.assertTrue(
            any(
                call.args and isinstance(call.args[0], dict)
                for call in eval_mock.mock_calls
            )
        )

    def test_sanitize_transformers_weights_maps_embedded_mimi_checkpoint(self):
        prefix = "audio_tokenizer."
        weights = {
            f"{prefix}encoder.layers.0.conv.weight": mx.arange(24).reshape(2, 3, 4),
            f"{prefix}decoder.layers.2.conv.weight": mx.arange(24).reshape(4, 2, 3),
            f"{prefix}upsample.conv.weight": mx.arange(24).reshape(4, 1, 6),
            f"{prefix}encoder_transformer.layers.0.self_attn.q_proj.weight": mx.full(
                (2, 2), 1
            ),
            f"{prefix}encoder_transformer.layers.0.self_attn.k_proj.weight": mx.full(
                (2, 2), 2
            ),
            f"{prefix}encoder_transformer.layers.0.self_attn.v_proj.weight": mx.full(
                (2, 2), 3
            ),
            f"{prefix}quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum": mx.ones(
                (8, 4)
            ),
            f"{prefix}quantizer.acoustic_residual_vector_quantizer.input_proj.weight": mx.arange(
                24
            ).reshape(
                2, 3, 4
            ),
            "text_model.layers.0.self_attn.q_proj.weight": mx.zeros((2, 2)),
        }

        converted = Mimi.sanitize_transformers_weights(weights, prefix=prefix)

        self.assertEqual(
            converted["encoder.init_conv1d.conv.conv.weight"].shape,
            (2, 4, 3),
        )
        self.assertEqual(
            converted["decoder.layers.0.upsample.convtr.convtr.weight"].shape,
            (2, 3, 4),
        )
        self.assertEqual(
            converted["upsample.convtr.convtr.convtr.weight"].shape,
            (4, 6, 1),
        )
        self.assertTrue(
            mx.array_equal(
                converted[
                    "encoder_transformer.transformer.layers.0.self_attn.in_proj.weight"
                ],
                mx.concatenate([mx.full((2, 2), value) for value in (1, 2, 3)], axis=0),
            )
        )
        self.assertIn(
            "quantizer.rvq_first.vq.layers.0.codebook.embedding_sum", converted
        )
        self.assertEqual(
            converted["quantizer.rvq_rest.input_proj.weight"].shape,
            (2, 4, 3),
        )
        self.assertNotIn("text_model.layers.0.self_attn.q_proj.weight", converted)

    def test_sanitize_transformers_weights_rejects_incomplete_qkv_group(self):
        with self.assertRaisesRegex(ValueError, "incomplete QKV group"):
            Mimi.sanitize_transformers_weights(
                {
                    "encoder_transformer.layers.0.self_attn.q_proj.weight": mx.ones(
                        (2, 2)
                    )
                }
            )

    def test_transformers_compatible_config_selects_matching_transformer_math(self):
        cfg = mimi_202407(32, transformers_compatible=True)

        self.assertFalse(cfg.transformer.rope_traditional)
        self.assertFalse(cfg.transformer.gelu_approximate)

        layer = MlpNoGating(cfg.transformer)
        inputs = mx.zeros((1, 1, cfg.transformer.d_model))
        with (
            patch(
                "mlx_audio.codec.models.mimi.modules.transformer.nn.gelu",
                side_effect=lambda value: value,
            ) as exact_gelu,
            patch(
                "mlx_audio.codec.models.mimi.modules.transformer.nn.gelu_approx",
                side_effect=lambda value: value,
            ) as approximate_gelu,
        ):
            layer(inputs)

        exact_gelu.assert_called_once()
        approximate_gelu.assert_not_called()

    def test_default_mimi_config_preserves_legacy_transformer_math(self):
        cfg = mimi_202407(32)

        self.assertTrue(cfg.transformer.rope_traditional)
        self.assertTrue(cfg.transformer.gelu_approximate)

    def test_load_transformers_weights_uses_sanitizer_and_finalizes_state(self):
        model = Mimi(mimi_202407(2, transformers_compatible=True))
        source = {"audio_tokenizer.source.weight": mx.ones((2, 2))}
        converted = {"converted.weight": mx.ones((2, 2))}

        with (
            patch.object(
                Mimi,
                "sanitize_transformers_weights",
                return_value=converted,
            ) as sanitize,
            patch.object(model, "load_weights", return_value=model) as load_weights,
            patch.object(model, "filter_and_map", return_value=model) as finalize,
        ):
            result = model.load_transformers_weights(
                source, prefix="audio_tokenizer.", strict=False
            )

        sanitize.assert_called_once_with(source, prefix="audio_tokenizer.")
        load_weights.assert_called_once_with(list(converted.items()), strict=False)
        finalize.assert_called_once()
        self.assertIs(result, model)

    def test_load_transformers_weights_rejects_legacy_transformer_math(self):
        model = Mimi(mimi_202407(2))

        with (
            patch.object(
                Mimi,
                "sanitize_transformers_weights",
                return_value={"converted.weight": mx.ones((2, 2))},
            ),
            patch.object(model, "load_weights", return_value=model),
        ):
            with self.assertRaisesRegex(ValueError, "transformers_compatible=True"):
                model.load_transformers_weights({"source.weight": mx.ones((2, 2))})


if __name__ == "__main__":
    unittest.main()
