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

fake_hf_hub = types.ModuleType("huggingface_hub")
fake_hf_hub.snapshot_download = lambda *_args, **_kwargs: None

fake_np = types.ModuleType("numpy")
fake_np.ndarray = type("ndarray", (), {})
fake_np.asarray = lambda value, *_args, **_kwargs: value

fake_tqdm = types.ModuleType("tqdm")

sys.modules.setdefault("mlx", fake_mlx)
sys.modules.setdefault("mlx.core", fake_mx)
sys.modules.setdefault("mlx.nn", fake_nn)
sys.modules.setdefault("mlx.utils", fake_utils)
sys.modules.setdefault("huggingface_hub", fake_hf_hub)
sys.modules.setdefault("numpy", fake_np)
sys.modules.setdefault("tqdm", fake_tqdm)
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
            assert get_model_category("wav2vec2", ["wav2vec2", "base"]) is None
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

    def test_sts_models_subpackages_remain_available_as_attributes(self):
        self.run_in_subprocess(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.sts as sts

            models = sts.models
            sentinel_modules = {
                "mlx_audio.sts.models.deepfilternet": SimpleNamespace(),
                "mlx_audio.sts.models.sam_audio": SimpleNamespace(),
                "mlx_audio.sts.models.lfm_audio": SimpleNamespace(),
                "mlx_audio.sts.models.mossformer2_se": SimpleNamespace(),
            }

            def import_side_effect(name, *args, **kwargs):
                if name in sentinel_modules:
                    return sentinel_modules[name]
                raise AssertionError(f"unexpected module import: {name}")

            with patch("mlx_audio.sts.models.importlib.import_module", side_effect=import_side_effect):
                assert models.deepfilternet is sentinel_modules["mlx_audio.sts.models.deepfilternet"]
                assert models.sam_audio is sentinel_modules["mlx_audio.sts.models.sam_audio"]
                assert models.lfm_audio is sentinel_modules["mlx_audio.sts.models.lfm_audio"]
                assert models.mossformer2_se is sentinel_modules["mlx_audio.sts.models.mossformer2_se"]

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

    def test_get_model_class_prefers_explicit_alias_over_unrelated_path_tokens(self):
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
                    model_type="fish_speech",
                    model_name=["spark", "fish", "audio", "s2", "pro"],
                    category="tts",
                    model_remapping=TTS_MODEL_REMAPPING,
                )

            assert arch is sentinel_module
            assert resolved_model_type == "fish_qwen3_omni"
            import_module.assert_called_once_with(
                "mlx_audio.tts.models.fish_qwen3_omni"
            )
            print("OK")
            """
        )

    def test_get_model_class_prefers_explicit_stt_alias_over_unrelated_path_tokens(self):
        self.run_in_subprocess(
            """
            from unittest.mock import patch

            from mlx_audio.model_routing import get_model_class
            from mlx_audio.stt.registry import MODEL_REMAPPING as STT_MODEL_REMAPPING

            sentinel_module = object()

            with patch(
                "mlx_audio.model_routing.importlib.import_module",
                return_value=sentinel_module,
            ) as import_module:
                arch, resolved_model_type = get_model_class(
                    model_type="vibevoice",
                    model_name=["sensevoice", "vibevoice", "small"],
                    category="stt",
                    model_remapping=STT_MODEL_REMAPPING,
                )

            assert arch is sentinel_module
            assert resolved_model_type == "vibevoice_asr"
            import_module.assert_called_once_with(
                "mlx_audio.stt.models.vibevoice_asr"
            )
            print("OK")
            """
        )

    def test_get_model_class_prefers_explicit_direct_model_type_over_path_tokens(self):
        self.run_in_subprocess(
            """
            from unittest.mock import patch

            from mlx_audio.model_routing import get_model_class
            from mlx_audio.stt.registry import MODEL_REMAPPING as STT_MODEL_REMAPPING

            sentinel_module = object()

            with patch(
                "mlx_audio.model_routing.importlib.import_module",
                return_value=sentinel_module,
            ) as import_module:
                arch, resolved_model_type = get_model_class(
                    model_type="whisper",
                    model_name=["custom", "moonshine", "checkpoint"],
                    category="stt",
                    model_remapping=STT_MODEL_REMAPPING,
                )

            assert arch is sentinel_module
            assert resolved_model_type == "whisper"
            import_module.assert_called_once_with("mlx_audio.stt.models.whisper")
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

    def test_public_top_level_loader_routes_generic_local_lid_wav2vec2_config_without_id2label(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "classifier_proj_size": 256,
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

            assert loaded == ("lid", "/tmp/generic-model"), loaded
            print("OK")
            """
        )

    def test_public_top_level_loader_does_not_treat_generic_wav2vec2_id2label_config_as_lid(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "id2label": {"0": "<pad>", "1": "a"},
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
                try:
                    utils.load_model("/tmp/generic-model")
                except ValueError as exc:
                    assert str(exc) == "Could not determine model type for /tmp/generic-model"
                else:
                    raise AssertionError(
                        "generic wav2vec2 config with only id2label should not route to lid"
                    )

            print("OK")
            """
        )

    def test_public_top_level_loader_loads_generic_wav2vec2_ctc_config_via_stt_runtime_routing(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils
            import mlx_audio.utils as utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config["model_type"],
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
            config = {
                "model_type": "wav2vec2",
                "architectures": ["Wav2Vec2ForCTC"],
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }
            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: dict(config),
                load_model=lambda model_name: ("tts", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/generic-model"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.utils._get_tts_utils",
                return_value=tts_utils,
            ), patch(
                "mlx_audio.utils._get_stt_utils",
                return_value=stt_utils,
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = utils.load_model("/tmp/generic-model")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] == "wav2vec2"
            assert model.config["model_path"] == "/tmp/generic-model"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_top_level_loader_loads_generic_wav2vec2_singular_architecture_ctc_config_via_stt_runtime_routing(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils
            import mlx_audio.utils as utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config["model_type"],
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
            config = {
                "model_type": "wav2vec2",
                "architecture": "Wav2Vec2ForCTC",
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }
            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: dict(config),
                load_model=lambda model_name: ("tts", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/generic-model"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.utils._get_tts_utils",
                return_value=tts_utils,
            ), patch(
                "mlx_audio.utils._get_stt_utils",
                return_value=stt_utils,
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = utils.load_model("/tmp/generic-model")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] == "wav2vec2"
            assert model.config["model_path"] == "/tmp/generic-model"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_top_level_loader_loads_architecture_only_wav2vec2_ctc_config_via_stt_runtime_routing(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils
            import mlx_audio.utils as utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config.get("model_type"),
                        "config_architecture": config.get("architecture"),
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
            config = {
                "architecture": "Wav2Vec2ForCTC",
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }
            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: dict(config),
                load_model=lambda model_name: ("tts", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/generic-model"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.utils._get_tts_utils",
                return_value=tts_utils,
            ), patch(
                "mlx_audio.utils._get_stt_utils",
                return_value=stt_utils,
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = utils.load_model("/tmp/generic-model")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] is None
            assert model.config["config_architecture"] == "Wav2Vec2ForCTC"
            assert model.config["model_path"] == "/tmp/generic-model"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_top_level_loader_prefers_ctc_architecture_over_stt_name_hints(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils
            import mlx_audio.utils as utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config["model_type"],
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
            config = {
                "model_type": "wav2vec2",
                "architectures": ["Wav2Vec2ForCTC"],
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }
            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: dict(config),
                load_model=lambda model_name: ("tts", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/custom-moonshine-checkpoint"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.utils._get_tts_utils",
                return_value=tts_utils,
            ), patch(
                "mlx_audio.utils._get_stt_utils",
                return_value=stt_utils,
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = utils.load_model("/tmp/custom-moonshine-checkpoint")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] == "wav2vec2"
            assert model.config["model_path"] == "/tmp/custom-moonshine-checkpoint"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_stt_loader_loads_architecture_only_wav2vec2_ctc_config_via_mms_runtime(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config.get("model_type"),
                        "config_architecture": config.get("architecture"),
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
            config = {
                "architecture": "Wav2Vec2ForCTC",
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/generic-model"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = stt_utils.load_model("/tmp/generic-model")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] is None
            assert model.config["config_architecture"] == "Wav2Vec2ForCTC"
            assert model.config["model_path"] == "/tmp/generic-model"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_top_level_loader_prefers_singular_ctc_architecture_over_stt_name_hints(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.stt.utils as stt_utils
            import mlx_audio.utils as utils

            class DummyModelConfig:
                @classmethod
                def from_dict(cls, config):
                    return {
                        "config_model_type": config["model_type"],
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
            config = {
                "model_type": "wav2vec2",
                "architecture": "Wav2Vec2ForCTC",
                "classifier_proj_size": 256,
                "id2label": {"0": "<pad>", "1": "a"},
            }
            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: dict(config),
                load_model=lambda model_name: ("tts", model_name),
            )
            lid_utils = SimpleNamespace(
                load_model=lambda model_name: ("lid", model_name),
            )
            vad_utils = SimpleNamespace(
                load_model=lambda model_name: ("vad", model_name),
            )

            def import_side_effect(name):
                if name == "mlx_audio.stt.models.mms":
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.get_model_path",
                return_value=Path("/tmp/custom-moonshine-checkpoint"),
            ), patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda _model_path: dict(config),
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.utils._get_tts_utils",
                return_value=tts_utils,
            ), patch(
                "mlx_audio.utils._get_stt_utils",
                return_value=stt_utils,
            ), patch("mlx_audio.utils._get_lid_utils", return_value=lid_utils), patch(
                "mlx_audio.utils._get_vad_utils", return_value=vad_utils
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                model = utils.load_model("/tmp/custom-moonshine-checkpoint")

            assert isinstance(model, DummyModel)
            assert model.config["config_model_type"] == "wav2vec2"
            assert model.config["model_path"] == "/tmp/custom-moonshine-checkpoint"
            assert model.strict is False
            assert len(model.loaded_weights) == 1
            assert model.loaded_weights[0][0] == "weight"
            assert model.eval_called is True
            import_module.assert_called_once_with("mlx_audio.stt.models.mms")
            print("OK")
            """
        )

    def test_public_top_level_loader_routes_mms_wav2vec2_ctc_config_to_stt(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "architectures": ["Wav2Vec2ForCTC"],
                    "classifier_proj_size": 256,
                    "id2label": {"0": "<pad>", "1": "a"},
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
                loaded = utils.load_model("/tmp/mms-generic-model")

            assert loaded == ("stt", "/tmp/mms-generic-model"), loaded
            print("OK")
            """
        )

    def test_public_top_level_loader_prefers_ctc_over_lid_looking_mms_path(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "architectures": ["Wav2Vec2ForCTC"],
                    "classifier_proj_size": 256,
                    "id2label": {"0": "<pad>", "1": "a"},
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
                loaded = utils.load_model("/tmp/mms-lid-256")

            assert loaded == ("stt", "/tmp/mms-lid-256"), loaded
            print("OK")
            """
        )

    def test_public_top_level_loader_prefers_lid_sequence_classification_over_mms_hint(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "wav2vec2",
                    "architectures": ["Wav2Vec2ForSequenceClassification"],
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
                loaded = utils.load_model("/tmp/mms-seqcls-generic-model")

            assert loaded == ("lid", "/tmp/mms-seqcls-generic-model"), loaded
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

    def test_public_tts_loader_resolves_documented_alias_model_types(self):
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
            configs = {
                "Ming-omni-tts-16.8B-A3B-bf16": {
                    "model_type": "ming_omni_tts",
                },
                "fish-audio-s2-pro": {
                    "model_type": "fish_speech",
                },
            }
            expected_module_paths = [
                "mlx_audio.tts.models.bailingmm",
                "mlx_audio.tts.models.fish_qwen3_omni",
            ]

            def import_side_effect(name):
                if name in expected_module_paths:
                    return dummy_module
                raise ImportError(f"missing {name}", name=name)

            with patch(
                "mlx_audio.utils.load_config",
                side_effect=lambda model_path: configs[Path(model_path).name],
            ), patch(
                "mlx_audio.utils.load_weights",
                return_value={"weight": object()},
            ), patch(
                "mlx_audio.utils.mx.eval",
            ), patch(
                "mlx_audio.model_routing.importlib.import_module",
                side_effect=import_side_effect,
            ) as import_module:
                ming = load_model(
                    Path("/tmp/Ming-omni-tts-16.8B-A3B-bf16"),
                    lazy=False,
                    strict=True,
                )
                fish = load_model(Path("/tmp/fish-audio-s2-pro"), lazy=False, strict=True)

            assert isinstance(ming, DummyModel)
            assert isinstance(fish, DummyModel)
            assert ming.config["resolved_model_type"] == "ming_omni_tts"
            assert fish.config["resolved_model_type"] == "fish_speech"
            assert ming.strict is True
            assert fish.strict is True
            assert ming.eval_called is True
            assert fish.eval_called is True
            assert [call.args[0] for call in import_module.call_args_list] == expected_module_paths
            print("OK")
            """
        )

    def test_public_top_level_loader_routes_documented_tts_model_types(self):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            configs = {
                "mlx-community/Ming-omni-tts-16.8B-A3B-bf16": {
                    "model_type": "ming_omni_tts",
                },
                "mlx-community/fish-audio-s2-pro": {
                    "model_type": "fish_speech",
                },
            }

            tts_utils = SimpleNamespace(
                load_config=lambda model_name: configs[model_name],
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
                ming = utils.load_model("mlx-community/Ming-omni-tts-16.8B-A3B-bf16")
                fish = utils.load_model("mlx-community/fish-audio-s2-pro")

            assert ming == ("tts", "mlx-community/Ming-omni-tts-16.8B-A3B-bf16")
            assert fish == ("tts", "mlx-community/fish-audio-s2-pro")
            print("OK")
            """
        )

    def test_public_top_level_loader_tolerates_null_architectures_for_resolved_model_type(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from types import SimpleNamespace
            from unittest.mock import patch

            import mlx_audio.utils as utils

            tts_utils = SimpleNamespace(
                load_config=lambda _model_name: {
                    "model_type": "qwen3_tts",
                    "architectures": None,
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
                loaded = utils.load_model("/tmp/qwen3-tts")

            assert loaded == ("tts", "/tmp/qwen3-tts"), loaded
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

    def test_public_sts_voice_pipeline_stays_optional_for_missing_compiled_extension_deps(
        self,
    ):
        self.run_in_subprocess_with_fake_mlx(
            """
            from unittest.mock import patch

            import mlx_audio.sts as sts

            real_import_module = sts.importlib.import_module

            def guarded_import(name, *args, **kwargs):
                if name == "mlx_audio.sts.voice_pipeline":
                    raise ImportError("blocked compiled extension", name="_webrtcvad")
                return real_import_module(name, *args, **kwargs)

            with patch("mlx_audio.sts.importlib.import_module", side_effect=guarded_import):
                assert sts.VoicePipeline is None

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
