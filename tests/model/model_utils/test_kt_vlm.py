from types import SimpleNamespace

import pytest

from llamafactory.model.model_utils import kt_vlm
from llamafactory.model.model_utils.visual import COMPOSITE_MODELS


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
    )
    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=True),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config=None),
    )
    kt_vlm.prepare_kt_vlm_conv3d(
        SimpleNamespace(use_kt=False),
        SimpleNamespace(model_type="qwen3_5_moe", vision_config={}),
    )

    assert calls == ["enabled"]


def test_validate_accepts_only_verified_modules(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)
    monkeypatch.setattr(
        kt_vlm,
        "_enable_compatibility",
        lambda: (_compatibility(), lambda: True, lambda model: ["visual.patch_embed.proj"]),
    )

    assert kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=True), object()) is True


def test_validate_fails_closed_when_patch_is_inactive(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)
    monkeypatch.setattr(
        kt_vlm,
        "_enable_compatibility",
        lambda: (_compatibility(), lambda: False, lambda model: ["visual.patch_embed.proj"]),
    )

    with pytest.raises(ValueError, match="did not replace"):
        kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=True), object())


def test_validate_preserves_non_kt_path(monkeypatch):
    monkeypatch.setattr(kt_vlm, "_is_torch_29", lambda: True)

    assert kt_vlm.validate_kt_vlm_conv3d(SimpleNamespace(use_kt=False), object()) is False


@pytest.mark.parametrize("model_type", ("qwen3_5", "qwen3_5_moe"))
def test_qwen35_language_side_lora_excludes_visual_tower(model_type):
    composite = COMPOSITE_MODELS[model_type]

    assert composite.vision_model_keys == ["visual"]
    assert "visual" in composite.lora_conflict_keys
