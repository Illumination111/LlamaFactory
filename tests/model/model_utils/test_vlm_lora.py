import pytest
import torch
from peft import LoraConfig, LoraModel
from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

from llamafactory.hparams import FinetuningArguments, ModelArguments
from llamafactory.model.model_utils.misc import find_all_linear_modules
from llamafactory.model.model_utils.visual import patch_target_modules
from llamafactory.model.model_utils.vlm_lora import find_vlm_lora_modules


class _VisionBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Linear(4, 4)


class _VisualTower(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, 4))
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv3d(3, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.blocks = torch.nn.ModuleList([_VisionBlock()])
        self.merger = torch.nn.Module()
        self.merger.linear_fc1 = torch.nn.Linear(4, 4)


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 4)
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].q_proj = torch.nn.Linear(4, 4)
        self.layers[0].conv1d = torch.nn.Conv1d(4, 4, kernel_size=3, padding=1, groups=4)


class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _VLMFixture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Config(model_type="qwen3_5_moe", tie_word_embeddings=False)
        self.model = torch.nn.Module()
        self.model.visual = _VisualTower()
        self.model.language_model = _LanguageModel()
        self.lm_head = torch.nn.Linear(4, 32)


@pytest.mark.parametrize(
    ("scope", "expected", "excluded"),
    (
        ("text", ("language_model.layers.0.q_proj",), ("visual", "embed_tokens", "lm_head")),
        (
            "vision",
            ("visual.patch_embed.proj", "visual.blocks.0.qkv", "visual.merger.linear_fc1"),
            ("language_model", "lm_head"),
        ),
        (
            "all",
            ("visual.patch_embed.proj", "visual.merger.linear_fc1", "language_model.layers.0.q_proj"),
            ("lm_head",),
        ),
    ),
)
def test_find_vlm_lora_modules_by_scope(scope, expected, excluded):
    targets = find_vlm_lora_modules(_VLMFixture(), scope)

    assert all(any(fragment in target for target in targets) for fragment in expected)
    assert all(all(fragment not in target for target in targets) for fragment in excluded)


@pytest.mark.parametrize("scope", ("text", "all"))
def test_find_vlm_lora_modules_excludes_grouped_conv(scope):
    targets = find_vlm_lora_modules(_VLMFixture(), scope)

    assert "model.language_model.layers.0.conv1d" not in targets


@pytest.mark.parametrize(
    ("scope", "freeze_state"),
    (
        ("text", (True, True, False)),
        ("vision", (False, False, True)),
        ("all", (False, False, False)),
    ),
)
def test_vlm_lora_scope_normalizes_freeze_flags(scope, freeze_state):
    args = FinetuningArguments(finetuning_type="lora", lora_target="all", vlm_lora_scope=scope)

    assert (
        args.freeze_vision_tower,
        args.freeze_multi_modal_projector,
        args.freeze_language_model,
    ) == freeze_state


@pytest.mark.parametrize("scope", ("default", "text", "all"))
def test_non_vision_scope_keeps_kt_lora_configuration(scope):
    finetuning_args = FinetuningArguments(
        finetuning_type="lora",
        lora_rank=4,
        lora_alpha=8,
        lora_target="all",
        vlm_lora_scope=scope,
    )
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_use_lora_experts=True)

    kt_config = model_args.get_kt_config_dict(finetuning_args, model_max_length=4096)

    assert kt_config["kt_lora_rank"] == 4
    assert kt_config["kt_lora_alpha"] == 8
    assert kt_config["kt_use_lora_experts"] is True
    assert kt_config["kt_freeze_experts"] is False


def test_vision_scope_disables_language_expert_lora_in_kt():
    finetuning_args = FinetuningArguments(
        finetuning_type="lora",
        lora_rank=4,
        lora_alpha=8,
        lora_target="all",
        vlm_lora_scope="vision",
    )
    model_args = ModelArguments(model_name_or_path="dummy", use_kt=True, kt_use_lora_experts=True)

    kt_config = model_args.get_kt_config_dict(finetuning_args, model_max_length=4096)

    assert kt_config["kt_lora_rank"] == 4
    assert kt_config["kt_lora_alpha"] == 8
    assert kt_config["kt_use_lora_experts"] is False
    assert kt_config["kt_freeze_experts"] is True


def test_default_vlm_lora_path_is_unchanged():
    model = _VLMFixture()
    args = FinetuningArguments(finetuning_type="lora", lora_target="all")

    targets = find_all_linear_modules(model, args.freeze_vision_tower)
    targets = patch_target_modules(model, args, targets)

    assert any("language_model.layers.0.q_proj" in target for target in targets)
    assert all("visual" not in target for target in targets)
    assert all("patch_embed" not in target for target in targets)
    assert all("merger" not in target for target in targets)


def test_vlm_lora_scope_rejects_custom_targets():
    with pytest.raises(ValueError, match="requires `lora_target=all`"):
        FinetuningArguments(finetuning_type="lora", lora_target="q_proj", vlm_lora_scope="vision")


def test_vlm_lora_scope_requires_composite_model():
    model = _VLMFixture()
    model.config.model_type = "qwen3_5_moe_text"

    with pytest.raises(ValueError, match="registered composite VLM"):
        find_vlm_lora_modules(model, "text")


@pytest.mark.parametrize(
    ("scope", "required_fragment", "forbidden_fragment"),
    (
        ("text", "language_model.layers.0.q_proj.lora_A", "visual"),
        ("vision", "visual.merger.linear_fc1.lora_A", "language_model"),
        ("all", "visual.blocks.0.qkv.lora_A", "not-present"),
    ),
)
def test_vlm_lora_scope_creates_only_requested_adapters(scope, required_fragment, forbidden_fragment):
    model = _VLMFixture()
    targets = find_vlm_lora_modules(model, scope)
    lora_model = LoraModel(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=targets),
        adapter_name="default",
    )
    trainable = [name for name, parameter in lora_model.named_parameters() if parameter.requires_grad]

    assert any(required_fragment in name for name in trainable)
    assert all(forbidden_fragment not in name for name in trainable)
    if scope == "all":
        assert any("visual.patch_embed.proj.lora_A" in name for name in trainable)
        assert any("language_model.layers.0.q_proj.lora_A" in name for name in trainable)


def test_all_scope_creates_trainable_conv3d_lora():
    model = _VLMFixture()
    targets = find_vlm_lora_modules(model, "all")
    lora_model = LoraModel(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=targets),
        adapter_name="default",
    )

    patch_embed = lora_model.model.model.visual.patch_embed.proj
    loss = patch_embed(torch.randn(1, 3, 2, 2, 2)).square().mean()
    loss.backward()

    assert patch_embed.lora_A["default"].weight.requires_grad
    assert patch_embed.lora_B["default"].weight.grad is not None
    assert patch_embed.lora_B["default"].weight.grad.ne(0).any()


def test_qwen35_moe_uses_real_text_vision_and_conv3d_targets():
    config = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_key_head_dim": 4,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "moe_intermediate_size": 4,
            "shared_expert_intermediate_size": 4,
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "max_position_embeddings": 64,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 1,
            "out_hidden_size": 8,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = Qwen3_5MoeForConditionalGeneration(config)

    text_targets = find_vlm_lora_modules(model, "text")
    vision_targets = find_vlm_lora_modules(model, "vision")
    all_targets = find_vlm_lora_modules(model, "all")

    assert any("model.language_model.layers.1.self_attn.q_proj" in target for target in text_targets)
    assert any("model.language_model.layers.0.linear_attn.in_proj_qkv" in target for target in text_targets)
    assert all("linear_attn.conv1d" not in target for target in text_targets)
    assert "model.visual.patch_embed.proj" in vision_targets
    assert any("model.visual.blocks.0.attn.qkv" in target for target in vision_targets)
    assert set(all_targets) == set(text_targets) | set(vision_targets)
