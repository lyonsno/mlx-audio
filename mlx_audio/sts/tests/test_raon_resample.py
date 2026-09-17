import importlib.util
import unittest
from pathlib import Path

import numpy as np

_PATH = Path(__file__).parents[1] / "models" / "raon" / "audio.py"
_SPEC = importlib.util.spec_from_file_location("raon_audio", _PATH)
_AUDIO = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIO)


class TestRaonResample(unittest.TestCase):
    def test_publisher_default_boundary_impulses(self):
        # Observed torchaudio 2.11.0 CPU FP32 functional.resample(24000, 16000).
        # Pinned Raon a8bac78f VoxtralWrapper._forward_streaming uses these defaults.
        audio = np.zeros(16, dtype=np.float32)
        audio[0], audio[-1] = 1, -1
        expected = np.array(
            [
                0.6600000262260437,
                0.006227732170373201,
                -0.0050268168561160564,
                0.00338068138808012,
                -0.0017213053070008755,
                0.0,
                0.0017213053070008755,
                -0.00338068138808012,
                0.0050268168561160564,
                -0.006227732170373201,
                -0.6600000262260437,
            ],
            dtype=np.float32,
        )
        actual = _AUDIO.resample_input_frame(audio)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=1e-6)

    def test_frame_shape_and_silence(self):
        actual = _AUDIO.resample_input_frame(np.zeros(1920, dtype=np.float32))
        np.testing.assert_array_equal(actual, np.zeros(1280, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
