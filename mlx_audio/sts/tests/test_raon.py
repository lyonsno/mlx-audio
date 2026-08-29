import unittest

import mlx.core as mx
import numpy as np

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


if __name__ == "__main__":
    unittest.main()
