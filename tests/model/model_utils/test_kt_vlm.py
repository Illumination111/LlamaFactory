from types import SimpleNamespace

import pytest

from llamafactory.extras import misc, packages
from llamafactory.model.model_utils import kt_vlm


KT_OPT_IN_ENV_VARS = (
    "ACCELERATE_USE_KT",
    "LLAMAFACTORY_ALLOW_TRANSFORMERS_KT",
    "USE_KT",
)


def _compatibility():
    return SimpleNamespace(required=True, active=True, torch_version="2.9.1", swift_version="4.4.2")


def test_prepare_activates_only_for_kt_vlm(monkeypatch):
    calls = []
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)
    monkeypatch.setattr(
        kt_vlm,
        "_enable_compatibility",
        lambda: (calls.append("enabled") or _compatibility(), lambda: True, lambda model: []),
    )

    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=True),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config={}),
        is_trainable=True,
    )
    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=True),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config=None),
        is_trainable=True,
    )
    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=False),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config={}),
        is_trainable=True,
    )
    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=True),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config={}),
        is_trainable=False,
    )

    assert calls == ["enabled"]


def test_validate_accepts_only_verified_modules(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)
    monkeypatch.setattr(
        kt_vlm,
        "_enable_compatibility",
        lambda: (_compatibility(), lambda: True, lambda model: ["visual.patch_embed.proj"]),
    )

    assert kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=True), object(), is_trainable=True) is True


def test_validate_fails_closed_when_patch_is_inactive(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)
    monkeypatch.setattr(
        kt_vlm,
        "_enable_compatibility",
        lambda: (_compatibility(), lambda: False, lambda model: ["visual.patch_embed.proj"]),
    )

    with pytest.raises(ValueError, match="did not replace"):
        kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=True), object(), is_trainable=True)


def test_validate_preserves_non_kt_path(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)

    assert kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=False), object(), is_trainable=True) is False
    assert kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=True), object(), is_trainable=False) is False


def test_verified_transformers_kt_requires_exact_versions_and_hooks(monkeypatch):
    versions = {"transformers": "5.6.0", "transformers-kt": "5.6.0.post1"}
    integration = SimpleNamespace(HfTrainerKTConfig=object(), is_kt_expert_loading_enabled=lambda: True)
    monkeypatch.setattr(packages.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(packages.importlib, "import_module", lambda _: integration)

    assert packages.is_verified_transformers_kt() is True

    versions["transformers-kt"] = "5.6.0.post2"
    assert packages.is_verified_transformers_kt() is False
    versions["transformers-kt"] = "5.6.0.post1"
    del integration.HfTrainerKTConfig
    assert packages.is_verified_transformers_kt() is False


@pytest.mark.parametrize(
    "opt_in_env",
    KT_OPT_IN_ENV_VARS,
)
def test_dependency_check_allows_only_opted_in_verified_transformers_kt(monkeypatch, opt_in_env):
    requirements = []
    for env_var in KT_OPT_IN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv(opt_in_env, "1")
    monkeypatch.setattr(misc, "is_verified_transformers_kt", lambda: True)
    monkeypatch.setattr(misc, "check_version", requirements.append)

    misc.check_dependencies()

    assert requirements[0] == "transformers>=4.55.0,<=5.8.0,!=4.57.0"
    assert requirements[1:] == [
        "datasets>=2.16.0,<=4.0.0",
        "accelerate>=1.3.0,<=1.15.0",
        "peft>=0.18.0,<=0.20.0",
        "trl>=0.18.0,<=0.24.0",
    ]


@pytest.mark.parametrize(("opt_in", "verified"), ((False, True), (True, False)))
def test_dependency_check_keeps_stock_transformers_exclusion(monkeypatch, opt_in, verified):
    requirements = []
    for env_var in KT_OPT_IN_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    if opt_in:
        monkeypatch.setenv("LLAMAFACTORY_ALLOW_TRANSFORMERS_KT", "1")

    monkeypatch.setattr(misc, "is_verified_transformers_kt", lambda: verified)
    monkeypatch.setattr(misc, "check_version", requirements.append)

    misc.check_dependencies()

    assert requirements[0] == "transformers>=4.55.0,<=5.8.0,!=4.57.0,!=5.6.0"
