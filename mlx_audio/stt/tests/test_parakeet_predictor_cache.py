"""Predictor reuse must preserve token decisions and recurrent-state commits."""

from dataclasses import asdict
from unittest.mock import Mock

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.parakeet.parakeet import Model


def make_model():
    return Model.from_config(
        {
            "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
            "model_defaults": {"tdt_durations": [0, 1, 2]},
            "preprocessor": {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": 8,
                "n_fft": 512,
                "dither": 0.0,
            },
            "encoder": {
                "feat_in": 8,
                "n_layers": 1,
                "d_model": 8,
                "n_heads": 2,
                "ff_expansion_factor": 2,
                "subsampling_factor": 2,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "conv_kernel_size": 3,
                "subsampling_conv_channels": 4,
                "pos_emb_max_len": 32,
            },
            "decoder": {
                "blank_as_pad": True,
                "vocab_size": 3,
                "prednet": {"pred_hidden": 8, "pred_rnn_layers": 2},
            },
            "joint": {
                "num_classes": 3,
                "vocabulary": ["a", "b", "<unk>"],
                "num_extra_outputs": 3,
                "jointnet": {
                    "joint_hidden": 8,
                    "activation": "relu",
                    "encoder_hidden": 8,
                    "pred_hidden": 8,
                },
            },
            "decoding": {
                "model_type": "tdt",
                "durations": [0, 1, 2],
                "greedy": {"max_symbols": 2},
            },
        }
    )


def scripted_decode(model, decisions, length, batch=1):
    features = mx.broadcast_to(
        mx.arange(length, dtype=mx.float32)[None, :, None], (batch, length, 8)
    )
    model.encoder = Mock(return_value=(features, mx.full((batch,), length)))
    recurrent = model.decoder.prediction["dec_rnn"]
    calls = Mock(wraps=recurrent)
    calls.num_layers = recurrent.num_layers
    calls.hidden_size = recurrent.hidden_size
    model.decoder.prediction["dec_rnn"] = calls
    frames = []
    predictions = []
    script = iter(decisions * batch)

    def joint(feature, prediction):
        frames.append(int(feature[0, 0, 0]))
        predictions.append(np.array(prediction))
        token, duration = next(script)
        logits = np.full((1, 1, 1, 7), -10.0, dtype=np.float32)
        logits[0, 0, 0, token] = 10
        logits[0, 0, 0, 4 + duration] = 10
        return mx.array(logits)

    model.joint = joint
    # Exercise actual control flow without tracing the scripted test decisions.
    model._compiled_tdt_step = model._tdt_step
    result = model.decode(mx.zeros((batch, length * 2, 8)))
    return result, calls, frames, predictions


def test_reuses_predictor_across_blanks_and_resets_per_batch():
    model = make_model()
    decisions = [(3, 1), (3, 1), (0, 0), (3, 2), (2, 1), (1, 1)]
    results, calls, frames, predictions = scripted_decode(model, decisions, 6, batch=2)
    assert [result.text for result in results] == ["ab", "ab"]
    assert frames == [0, 1, 2, 2, 4, 5] * 2
    assert calls.call_count == 6  # Initial blank, then each committed token.
    assert asdict(results[0]) == asdict(results[1])
    for offset in (0, 6):
        np.testing.assert_array_equal(predictions[offset], predictions[offset + 1])
        np.testing.assert_array_equal(predictions[offset], predictions[offset + 2])
    np.testing.assert_array_equal(calls.call_args_list[0].args[1][0], 0)
    np.testing.assert_array_equal(calls.call_args_list[3].args[1][0], 0)
    tokens = results[0].sentences[0].tokens
    assert [token.id for token in tokens] == [0, 1]
    assert [token.start for token in tokens] == pytest.approx([0.04, 0.1])
    assert [token.duration for token in tokens] == pytest.approx([0, 0.02])


def test_blank_zero_duration_symbol_limit_does_not_invalidate_predictor():
    model = make_model()
    results, calls, frames, _ = scripted_decode(model, [(3, 0)] * 6, 3)
    assert results[0].text == ""
    assert frames == [0, 0, 1, 1, 2, 2]
    assert calls.call_count == 1


def test_repeated_nonblank_token_still_advances_predictor():
    model = make_model()
    results, calls, frames, predictions = scripted_decode(
        model, [(0, 0), (0, 0), (3, 1)], 2
    )
    assert results[0].text == "aa"
    assert frames == [0, 0, 1]
    assert calls.call_count == 3
    assert not np.array_equal(predictions[0], predictions[1])
    assert not np.array_equal(predictions[1], predictions[2])


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("token", [0, 3])
def test_compiled_cached_step_matches_recomputed_step(dtype, token):
    mx.random.seed(7)
    model = make_model()
    hidden, cell = model._make_initial_decoder_state(1, dtype)
    current_token = mx.array([[token]])
    first = mx.random.normal((1, 1, 8)).astype(dtype)
    second = mx.random.normal((1, 1, 8)).astype(dtype)
    _, _, next_hidden, next_cell, output = model._compiled_tdt_step(
        first, current_token, hidden, cell
    )
    cache = (output, next_hidden, next_cell)
    cached = model._compiled_tdt_step(second, current_token, hidden, cell, cache)
    recomputed = model._compiled_tdt_step(second, current_token, hidden, cell)
    mx.eval(cached, recomputed)
    for actual, expected in zip(cached, recomputed):
        np.testing.assert_array_equal(
            actual.astype(mx.float32), expected.astype(mx.float32)
        )
