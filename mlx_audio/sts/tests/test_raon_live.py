import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class TestLiveAudioExchange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[3] / "examples" / "raon_live.py"
        if not path.is_file():
            raise AssertionError("Live RAON PCM exchange has not been implemented")
        spec = importlib.util.spec_from_file_location("raon_live_demo", path)
        cls.demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.demo)

    def callback(self, exchange, values, status=""):
        source = np.asarray(values, dtype=np.float32)[:, None]
        target = np.full_like(source, 99)
        clock = SimpleNamespace(inputBufferAdcTime=1.0, outputBufferDacTime=2.0)
        exchange.callback(source, target, len(source), clock, status)
        return source, target

    def test_empty_playback_is_silence_and_capture_continues(self):
        exchange = self.demo.AudioExchange()
        source, target = self.callback(exchange, [1, 2, 3])
        np.testing.assert_array_equal(target[:, 0], [0, 0, 0])
        np.testing.assert_array_equal(exchange.inputs.get_nowait(), source[:, 0])
        self.assertEqual(exchange.underflow_samples, 3)

    def test_partial_output_and_multiple_chunks_preserve_order(self):
        exchange = self.demo.AudioExchange()
        exchange.push_audio(np.array([1, 2, 3], dtype=np.float32))
        exchange.push_audio(np.array([4, 5], dtype=np.float32))
        _, first = self.callback(exchange, [0, 0])
        _, second = self.callback(exchange, [0, 0, 0, 0])
        np.testing.assert_array_equal(first[:, 0], [1, 2])
        np.testing.assert_array_equal(second[:, 0], [3, 4, 5, 0])

    def test_recording_contains_copied_microphone_and_actual_playback(self):
        exchange = self.demo.AudioExchange()
        exchange.push_audio(np.array([0.5, 0.25], dtype=np.float32))
        source, target = self.callback(exchange, [1, 2, 3], "input overflow")
        source.fill(0)
        target.fill(0)
        recording, metadata = exchange.recordings.get_nowait()
        np.testing.assert_array_equal(recording[:, 0], [1, 2, 3])
        np.testing.assert_array_equal(recording[:, 1], [0.5, 0.25, 0])
        self.assertEqual(metadata["status"], "input overflow")
        self.assertEqual(metadata["input_adc_time"], 1.0)
        self.assertEqual(metadata["output_dac_time"], 2.0)

    def test_pending_input_is_retained_without_queue_eviction(self):
        exchange = self.demo.AudioExchange()
        for index in range(100):
            self.callback(exchange, [index])
        self.assertEqual(exchange.inputs.qsize(), 100)
        self.assertEqual(exchange.recordings.qsize(), 100)
        for index in range(100):
            self.assertEqual(exchange.inputs.get_nowait().item(), index)

    def test_nonfinite_or_multichannel_generated_audio_is_rejected(self):
        exchange = self.demo.AudioExchange()
        for values in (np.array([np.nan]), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                exchange.push_audio(values)

    def test_model_failure_closes_stream_and_preserves_captured_audio(self):
        from scipy.io import wavfile

        finished = threading.Event()
        stream_closed = []

        class Stream:
            samplerate = 24_000
            device = (0, 1)
            latency = (0.08, 0.08)

            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]

            def __enter__(self):
                clock = SimpleNamespace(inputBufferAdcTime=1.0, outputBufferDacTime=2.0)
                self.callback(
                    np.full((1920, 1), 0.25, dtype=np.float32),
                    np.zeros((1920, 1), dtype=np.float32),
                    1920,
                    clock,
                    "",
                )
                return self

            def __exit__(self, *args):
                stream_closed.append(True)

        class Session:
            def step(self, frame):
                finished.set()
                raise RuntimeError("injected model failure")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                patch("sounddevice.Stream", Stream),
                patch("builtins.input", side_effect=lambda: finished.wait()),
            ):
                with self.assertRaisesRegex(RuntimeError, "injected model failure"):
                    self.demo.converse(
                        Session(), directory, 0, 1, directory / "live.json"
                    )
            report = json.loads((directory / "status.json").read_text())
            rate, audio = wavfile.read(directory / "microphone-left-playback-right.wav")
            self.assertEqual(report["state"], "failed")
            self.assertEqual(report["captured_samples"], 1920)
            self.assertEqual(rate, 24000)
            np.testing.assert_array_equal(audio[:, 0], np.full(1920, 0.25))
            np.testing.assert_array_equal(audio[:, 1], np.zeros(1920))
        self.assertEqual(stream_closed, [True])


if __name__ == "__main__":
    unittest.main()
