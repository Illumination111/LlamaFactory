# KT VLM LoRA scopes

The optional `vlm_lora_scope` setting provides three explicit LoRA target scopes for registered composite VLMs:

| Value | Trainable LoRA modules | KT offloaded language experts |
| --- | --- | --- |
| `text` | Language-side Linear/Conv modules | LoRA enabled |
| `vision` | Vision tower and multimodal projector Linear/Conv modules, including Conv3D patch embedding | Frozen; KT uses its input-gradient-only backend |
| `all` | Union of text and vision targets | LoRA enabled |

These scopes select LoRA adapters across the requested modalities; they do not turn base model weights into trainable full-FT parameters. `lora_target: all` is required with every non-default scope.

Existing configurations are unchanged because the default is `vlm_lora_scope: default`. The Conv3D compatibility layer is also opt-in: it is activated only for trainable Qwen VLM runs with `use_kt: true` on PyTorch 2.9.x.

Example launches for Qwen3.5-122B-A10B and the built-in `mllm_demo` dataset:

```bash
llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_text_lora_sft_kt.yaml
llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_vision_lora_sft_kt.yaml
llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml
```

On PyTorch 2.9.x, install a KT kernel build with the optional compatibility dependency before launching:

```bash
pip install 'kt-kernel[vlm-sft]'
```
