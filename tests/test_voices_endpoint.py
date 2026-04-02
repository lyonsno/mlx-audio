"""Tests for the /v1/voices endpoint and _discover_voices function."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@dataclass
class FakeConfig:
    model_path: str = ""


class FakeVoxtralModel:
    """Simulates a voxtral_tts model with _voice_embedding_files."""

    def __init__(self, voices):
        self._voice_embedding_files = {v: Path(f"/fake/{v}.safetensors") for v in voices}


class FakeVibeVoiceModel:
    """Simulates a vibevoice model with config.model_path + voices/ dir."""

    def __init__(self, model_path):
        self.config = FakeConfig(model_path=str(model_path))


class FakeQwen3TTSModel:
    """Simulates a qwen3_tts model with get_supported_speakers()."""

    def __init__(self, speakers):
        self._speakers = speakers

    def get_supported_speakers(self):
        return self._speakers


class FakeKokoroModel:
    """Simulates a kokoro model with repo_id."""

    def __init__(self, repo_id):
        self.repo_id = repo_id


class FakeUnknownModel:
    """A model with no recognizable voice pattern."""

    pass


@pytest.fixture
def discover_voices():
    from mlx_audio.server import _discover_voices

    return _discover_voices


class TestDiscoverVoices:
    def test_voxtral_returns_embedding_file_keys(self, discover_voices):
        model = FakeVoxtralModel(["casual_female", "neutral_male", "cheerful_female"])
        voices = discover_voices(model)
        assert voices == ["casual_female", "cheerful_female", "neutral_male"]

    def test_voxtral_empty(self, discover_voices):
        model = FakeVoxtralModel([])
        # Falls through to other methods, but none match either
        voices = discover_voices(model)
        assert voices == []

    def test_vibevoice_scans_voices_dir(self, discover_voices):
        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            (voices_dir / "en-Emma_woman.safetensors").touch()
            (voices_dir / "en-Carter_man.safetensors").touch()
            (voices_dir / "fr-Spk0_man.safetensors").touch()
            # Non-safetensors files should be ignored
            (voices_dir / "README.md").touch()

            model = FakeVibeVoiceModel(tmpdir)
            voices = discover_voices(model)
            assert voices == ["en-Carter_man", "en-Emma_woman", "fr-Spk0_man"]

    def test_vibevoice_no_voices_dir(self, discover_voices):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = FakeVibeVoiceModel(tmpdir)
            voices = discover_voices(model)
            assert voices == []

    def test_qwen3_tts_returns_speakers(self, discover_voices):
        model = FakeQwen3TTSModel(["Vivian", "Ryan", "Aria"])
        voices = discover_voices(model)
        assert voices == ["Aria", "Ryan", "Vivian"]

    def test_qwen3_tts_empty_speakers(self, discover_voices):
        model = FakeQwen3TTSModel([])
        voices = discover_voices(model)
        assert voices == []

    def test_kokoro_scans_hf_cache_via_repo_id(self, discover_voices):
        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            (voices_dir / "af_heart.safetensors").touch()
            (voices_dir / "af_bella.safetensors").touch()
            (voices_dir / "af_heart.pt").touch()  # non-safetensors ignored

            model = FakeKokoroModel(repo_id="mlx-community/Kokoro-82M-bf16")
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(tmpdir)
            ):
                voices = discover_voices(model)
            assert voices == ["af_bella", "af_heart"]

    def test_hf_cache_fallback_via_model_name(self, discover_voices):
        """When model has no repo_id, model_name is used for HF cache lookup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            voices_dir = Path(tmpdir) / "voices"
            voices_dir.mkdir()
            (voices_dir / "voice_a.safetensors").touch()

            model = FakeUnknownModel()
            with patch(
                "huggingface_hub.snapshot_download", return_value=str(tmpdir)
            ):
                voices = discover_voices(model, model_name="some/repo")
            assert voices == ["voice_a"]

    def test_unknown_model_returns_empty(self, discover_voices):
        model = FakeUnknownModel()
        voices = discover_voices(model)
        assert voices == []

    def test_deduplicates_voices(self, discover_voices):
        """If the same voice appears via multiple discovery paths, deduplicate."""
        model = FakeVoxtralModel(["voice_a", "voice_b", "voice_a"])
        voices = discover_voices(model)
        assert voices == ["voice_a", "voice_b"]


class TestVoicesEndpoint:
    @pytest.fixture
    def client(self):
        from mlx_audio.server import app, model_provider

        model_provider.models.clear()
        model_provider.models["test-voxtral"] = FakeVoxtralModel(
            ["casual_female", "neutral_male"]
        )
        model_provider.models["test-qwen3"] = FakeQwen3TTSModel(
            ["Vivian", "Ryan"]
        )
        return TestClient(app)

    def test_list_all_voices(self, client):
        resp = client.get("/v1/voices")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 2

        by_model = {d["model"]: d["voices"] for d in data["data"]}
        assert by_model["test-voxtral"] == ["casual_female", "neutral_male"]
        assert by_model["test-qwen3"] == ["Ryan", "Vivian"]

    def test_filter_by_model(self, client):
        resp = client.get("/v1/voices?model_name=test-voxtral")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["model"] == "test-voxtral"
        assert data["data"][0]["voices"] == ["casual_female", "neutral_male"]

    def test_filter_model_not_found(self, client):
        resp = client.get("/v1/voices?model_name=nonexistent")
        assert resp.status_code == 404
