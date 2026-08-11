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

from typing import TYPE_CHECKING, Literal

import torch

from ...extras import logging
from .visual import COMPOSITE_MODELS


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger(__name__)

VlmLoraScope = Literal["text", "vision", "all"]

_CONV_TYPES = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)


def _is_supported_lora_module(module: torch.nn.Module) -> bool:
    """Match modules that are safe for automatic PEFT LoRA discovery."""
    if isinstance(module, torch.nn.Linear):
        return True

    # PEFT requires the LoRA rank to be divisible by groups and cannot merge
    # grouped-convolution adapters. Keep automatic discovery rank-independent
    # and safe while retaining regular vision patch-embedding convolutions.
    if isinstance(module, _CONV_TYPES):
        return module.groups == 1

    # Preserve support for quantized/custom Linear implementations used by the
    # existing `find_all_linear_modules` path.
    class_name = module.__class__.__name__
    return "Linear" in class_name and "Embedding" not in class_name


def find_vlm_lora_modules(model: "PreTrainedModel", scope: VlmLoraScope) -> list[str]:
    """Return exact PEFT targets for one or both sides of a composite VLM."""
    model_type = getattr(model.config, "model_type", None)
    if model_type not in COMPOSITE_MODELS:
        raise ValueError(
            f"`vlm_lora_scope={scope}` requires a registered composite VLM, got model_type={model_type!r}."
        )

    composite = COMPOSITE_MODELS[model_type]
    text_keys = composite.language_model_keys
    vision_keys = composite.vision_model_keys + composite.projector_keys
    if scope == "text":
        selected_keys = text_keys
    elif scope == "vision":
        selected_keys = vision_keys
    elif scope == "all":
        selected_keys = text_keys + vision_keys
    else:
        raise ValueError(f"Unknown VLM LoRA scope: {scope!r}.")

    target_modules = []
    for name, module in model.named_modules():
        if not name or "lm_head" in name:
            continue
        if any(key in name for key in selected_keys) and _is_supported_lora_module(module):
            target_modules.append(name)

    if not target_modules:
        raise ValueError(f"No PEFT-compatible modules were found for `vlm_lora_scope={scope}`.")

    logger.info_rank0(f"Found {len(target_modules)} VLM LoRA modules for scope `{scope}`.")
    return target_modules
