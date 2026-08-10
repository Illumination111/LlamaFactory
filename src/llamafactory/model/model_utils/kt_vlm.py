# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

from ...extras import logging
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)

_KT_VLM_CONV3D_MODEL_TYPES = {
    "qwen2_vl",
    "qwen2_5_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_5",
    "qwen3_5_moe",
}
_KT_VLM_INSTALL_HINT = "Install it with `pip install 'kt-kernel[vlm-sft]'`."


def _is_torch_29() -> bool:
    return is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0")


def _is_supported_vlm_config(config: "PretrainedConfig") -> bool:
    """Identify known Qwen VLM configs without matching text-only Qwen models."""
    return (
        getattr(config, "model_type", None) in _KT_VLM_CONV3D_MODEL_TYPES
        and getattr(config, "vision_config", None) is not None
    )


def _load_compatibility_api():
    try:
        from kt_kernel.sft.conv3d_compat import (
            enable_swift_conv3d_patch,
            is_swift_conv3d_patch_active,
            validate_swift_conv3d_modules,
        )
    except ImportError as exc:
        raise RuntimeError(f"KT VLM Conv3D compatibility API is unavailable. {_KT_VLM_INSTALL_HINT}") from exc

    return enable_swift_conv3d_patch, is_swift_conv3d_patch_active, validate_swift_conv3d_modules


def _enable_compatibility() -> tuple[Any, Any, Any]:
    enable_patch, is_patch_active, validate_modules = _load_compatibility_api()
    try:
        compatibility = enable_patch()
    except RuntimeError as exc:
        raise RuntimeError(f"failed to enable the KT VLM Conv3D compatibility layer: {exc}") from exc

    if compatibility.required and (not compatibility.active or not is_patch_active()):
        raise RuntimeError("the KT VLM Conv3D compatibility layer did not become active in this rank")

    return compatibility, is_patch_active, validate_modules


def prepare_kt_vlm_conv3d(model_args: "ModelArguments", config: "PretrainedConfig", is_trainable: bool) -> None:
    """Activate compatibility before weight loading for a known KT VLM."""
    if not is_trainable or not model_args.use_kt or not _is_torch_29() or not _is_supported_vlm_config(config):
        return

    try:
        compatibility, _, _ = _enable_compatibility()
    except RuntimeError as exc:
        raise ValueError(f"Cannot prepare KT VLM training on torch 2.9.x: {exc}. {_KT_VLM_INSTALL_HINT}") from exc

    logger.info_rank0(
        "Enabled KT VLM Conv3D compatibility before model loading: "
        f"torch={compatibility.torch_version}, ms-swift={compatibility.swift_version}."
    )


def validate_kt_vlm_conv3d(model_args: "ModelArguments", model: "PreTrainedModel", is_trainable: bool) -> bool:
    """Return whether KT safely handles the model's torch 2.9 Conv3D modules."""
    if not is_trainable or not model_args.use_kt or not _is_torch_29():
        return False

    try:
        compatibility, is_patch_active, validate_modules = _enable_compatibility()
        module_names = validate_modules(model)
        if not module_names:
            return False
        if not is_patch_active():
            raise RuntimeError("ms-swift did not replace torch.nn.Conv3d.forward in this rank")
    except RuntimeError as exc:
        raise ValueError(f"KT cannot safely run this VLM's Conv3D modules: {exc}. {_KT_VLM_INSTALL_HINT}") from exc

    logger.warning_rank0(
        "Using the verified KT/ms-swift Conv3D replacement for torch 2.9.x: "
        f"modules={module_names}, ms-swift={compatibility.swift_version}."
    )
    return True
