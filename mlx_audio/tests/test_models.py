"""Repo-level structural smoke tests for lightweight routing and package seams."""

import subprocess
import sys
import textwrap
import unittest

FAKE_MLX_PREAMBLE = """
import sys
import types

fake_mx = types.ModuleType("mlx.core")
fake_mx.Dtype = type("Dtype", (), {})
fake_mx.float32 = object()
fake_mx.eval = lambda *_args, **_kwargs: None
fake_mx.load = lambda *_args, **_kwargs: {}
fake_mx.array = type("array", (), {})

fake_nn = types.ModuleType("mlx.nn")
fake_nn.Module = type("Module", (), {})
fake_nn.quantize = lambda *_args, **_kwargs: None

fake_utils = types.ModuleType("mlx.utils")
fake_utils.tree_flatten = lambda tree: tree

fake_mlx = types.ModuleType("mlx")
fake_mlx.core = fake_mx
fake_mlx.nn = fake_nn
fake_mlx.utils = fake_utils

sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_mx)
sys.modules.setdefault("mlx.nn", fake_nn)
sys.modules.setdefault("mlx.utils", fake_utils)
"""


class SmokeSubprocessTestCase(unittest.TestCase):
    def run_in_subprocess(self, code: str):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def run_in_subprocess_with_fake_mlx(self, code: str):
        self.run_in_subprocess(
            textwrap.dedent(FAKE_MLX_PREAMBLE) + "\n" + textwrap.dedent(code)
        )


class TestModelCategoryRouting(SmokeSubprocessTestCase):
    def test_get_model_category_avoids_unrelated_optional_imports(self):
        self.run_in_subprocess(
            """
            import builtins

            real_import = builtins.__import__
            blocked_prefixes = (
                "scipy",
                "torch",
                "mlx_audio.stt.utils",
                "mlx_audio.stt.models",
                "mlx_audio.tts.utils",
                "mlx_audio.vad.utils",
                "mlx_audio.lid.utils",
                "mlx_audio.sts.voice_pipeline",
            )

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.startswith(blocked_prefixes):
                    raise AssertionError(
                        f"unexpected import during category detection: {name}"
                    )
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            from mlx_audio.model_routing import get_model_category

            assert get_model_category("qwen3_tts", ["qwen3", "tts"]) == "tts"
            assert get_model_category("sensevoice", ["sensevoice"]) == "stt"
            assert get_model_category("smart_turn", ["smart", "turn"]) == "vad"
            assert get_model_category("ecapa_tdnn", ["ecapa", "tdnn"]) == "lid"
            assert get_model_category("vibevoice", ["vibevoice", "asr"]) == "stt"
            assert get_model_category("wav2vec2", ["mms", "lid", "256"]) == "lid"
            assert get_model_category("wav2vec2", ["wav2vec2", "base"]) != "lid"
            assert get_model_category("definitely_unknown_model", ["unknown"]) is None
            print("OK")
            """
        )


class TestPackageCompatibility(SmokeSubprocessTestCase):
    def test_category_packages_still_expose_utils_submodule(self):
        self.run_in_subprocess(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.lid as lid
            import mlx_audio.stt as stt
            import mlx_audio.tts as tts
            import mlx_audio.vad as vad

            lid_utils = SimpleNamespace(__name__="mlx_audio.lid.utils")
            stt_utils = SimpleNamespace(__name__="mlx_audio.stt.utils")
            tts_utils = SimpleNamespace(__name__="mlx_audio.tts.utils")
            vad_utils = SimpleNamespace(__name__="mlx_audio.vad.utils")

            with patch("mlx_audio.lid.importlib.import_module", return_value=lid_utils):
                assert lid.utils is lid_utils

            with patch("mlx_audio.stt.importlib.import_module", return_value=stt_utils):
                assert stt.utils is stt_utils

            with patch("mlx_audio.tts.importlib.import_module", return_value=tts_utils):
                assert tts.utils is tts_utils

            with patch("mlx_audio.vad.importlib.import_module", return_value=vad_utils):
                assert vad.utils is vad_utils

            print("OK")
            """
        )

    def test_sts_package_still_exposes_models_submodule(self):
        self.run_in_subprocess(
            """
            import mlx_audio.sts as sts

            assert sts.models.__name__ == "mlx_audio.sts.models"
            print("OK")
            """
        )


class TestModelClassRouting(SmokeSubprocessTestCase):
    def test_get_model_class_resolves_representative_tts_family(self):
        self.run_in_subprocess(
            """
            from unittest.mock import patch

            from mlx_audio.model_routing import get_model_class
            from mlx_audio.tts.registry import MODEL_REMAPPING as TTS_MODEL_REMAPPING

            sentinel_module = object()

            with patch(
                "mlx_audio.model_routing.importlib.import_module",
                return_value=sentinel_module,
            ) as import_module:
                arch, resolved_model_type = get_model_class(
                    model_type="qwen3_tts",
                    model_name=["qwen3", "tts"],
                    category="tts",
                    model_remapping=TTS_MODEL_REMAPPING,
                )

            assert arch is sentinel_module
            assert resolved_model_type == "qwen3_tts"
            import_module.assert_called_once_with("mlx_audio.tts.models.qwen3_tts")
            print("OK")
            """
        )

    def test_public_top_level_loader_routes_generic_local_lid_wav2vec2_config(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "classifier_proj_size": 256,
                    "id2label": {"en": "English"},
                },
                load_model=lambda model_name: ("tts", model_name),
            )
            stt_utils = SimpleNamespace(
                load_model=lambda model_name: ("stt", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            with patch("mlx_audio.utils._get_tts_utils", return_value=tts_utils), patch(
                "mlx_audio.utils._get_stt_utils", return_value=stt_utils
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ):
                loaded = utils.load_model("/tmp/generic-model")

            assert loaded == ("lid", "/tmp/generic-model")
            print("OK")
            """
        )

    def test_public_tts_loader_uses_shared_runtime_routing(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            from mlx_audio.tts.utils import load_model

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "resolved_model_type": config["model_type"],
                        "model_path": config["model_path"],
                    }

            class DummyModel:
                def __init__(self, config):
                    self.config = config
                    self.loaded_weights = None
                    self.strict = None
                    self.eval_called = False

                def load_weights(self, items, strict=False):
                    self.loaded_weights = list(items)
                    self.strict = strict

                def parameters(self):
                    return ()

                def eval(self):
                    self.eval_called = True

            dummy_module = SimpleNamespace(
                ModelConfig=DummyModelConfig,
                Model=DummyModel,
            )

            with patch(
                "mlx_audio.utils.load_config",
                return_value={"model_type": "qwen3_tts"},
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                return_value=dummy_module,
            ) as import_module:
                model = load_model(Path("/tmp/qwen3-tts"), lazy=False, strict=True)

            assert isinstance(model, DummyModel)
            assert model.config["resolved_model_type"] == "qwen3_tts"
            assert model.config["model_path"] == "/tmp/qwen3-tts"
            assert model.strict is True
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.tts.models.qwen3_tts")
            print("OK")
            """
        )


class TestPublicStsStructuralSmoke(SmokeSubprocessTestCase):
    def test_public_sts_voice_pipeline_stays_optional_for_missing_realtime_deps(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            import builtins

            real_import = builtins.__import__
            blocked_prefixes = (
                "sounddevice",
                "webrtcvad",
                "mlx_lm",
            )

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.startswith(blocked_prefixes):
                    raise ImportError(f"blocked {name}", name=name)
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            from mlx_audio.sts import VoicePipeline

            assert VoicePipeline is None
            print("OK")
            """
        )

    def test_public_sts_voice_pipeline_stays_optional_for_missing_pkg_resources(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            import builtins

            real_import = builtins.__import__

            def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name.startswith("pkg_resources"):
                    raise ImportError(f"blocked {name}", name=name)
                return real_import(name, globals, locals, fromlist, level)

            builtins.__import__ = guarded_import

            from mlx_audio.sts import VoicePipeline

            assert VoicePipeline is None
            print("OK")
            """
        )

    def test_public_sts_voice_pipeline_propagates_non_optional_import_errors(self):
        self.run_in_subprocess(
            """
            from unittest.mock import patch

            import mlx_audio.sts as sts

            real_import_module = sts.importlib.import_module

            def guarded_import(name, *args, **kwargs):
                if name == "mlx_audio.sts.voice_pipeline":
                    raise ImportError("unexpected regression", name=name)
                return real_import_module(name, *args, **kwargs)

            with patch("mlx_audio.sts.importlib.import_module", side_effect=guarded_import):
                try:
                    _ = sts.VoicePipeline
                except ImportError as exc:
                    assert exc.name == "mlx_audio.sts.voice_pipeline"
                else:
                    raise AssertionError("VoicePipeline should propagate non-optional ImportError")

            print("OK")
            """
        )

    def test_sts_models_deepfilternet_import_avoids_sibling_subsystems(self):
        self.run_in_subprocess(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.sts as sts

            models = sts.models
            sentinel_module = SimpleNamespace(
                DeepFilterNet3Config=type("DeepFilterNet3Config", (), {}),
                DeepFilterNetModel=type("DeepFilterNetModel", (), {}),
            )

            blocked_prefixes = (
                "mlx_audio.sts.models.lfm_audio",
                "mlx_audio.sts.models.mossformer2_se",
                "mlx_audio.sts.models.sam_audio",
                "mlx_audio.sts.voice_pipeline",
            )

            def guarded_import(name, *args, **kwargs):
                if name.startswith(blocked_prefixes):
                    raise AssertionError(
                        f"unexpected import during public STS smoke: {name}"
                    )
                if name == "mlx_audio.sts.models.deepfilternet":
                    return sentinel_module
                raise AssertionError(f"unexpected module import: {name}")

            with patch("mlx_audio.sts.models.importlib.import_module", side_effect=guarded_import):
                assert models.DeepFilterNet3Config is sentinel_module.DeepFilterNet3Config
                assert models.DeepFilterNetModel is sentinel_module.DeepFilterNetModel

            print("OK")
            """
        )


if __name__ == "__main__":
    unittest.main()
