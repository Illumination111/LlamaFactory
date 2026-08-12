# KT VLM LoRA scopes

The optional `vlm_lora_scope` setting provides three explicit LoRA target scopes for registered composite VLMs:

| Value | Trainable LoRA modules | KT offloaded language experts |
| --- | --- | --- |
| `text` | Language-side Linear/Conv modules | LoRA enabled |
| `vision` | Vision tower and multimodal projector Linear/Conv modules, including Conv3D patch embedding | Frozen; KT uses its input-gradient-only backend |
| `all` | Union of text and vision targets | LoRA enabled |

These scopes select LoRA adapters across the requested modalities; they do not turn base model weights into trainable full-FT parameters. `lora_target: all` is required with every non-default scope.

Existing configurations are unchanged because the default is `vlm_lora_scope: default`. The Conv3D compatibility layer is also opt-in: it is activated only for trainable Qwen VLM runs with `use_kt: true` on PyTorch 2.9.x.

The KT SFT stack currently provides `transformers-kt==5.6.0.post1`, whose import
package reports `transformers==5.6.0`. LlamaFactory keeps rejecting the broken
stock 5.6.0 release, but accepts this exact KT fork when KT is explicitly
enabled and its integration hooks are present. `accelerate launch` sets
`ACCELERATE_USE_KT=true` from a configuration containing an enabled
`kt_config`; for a direct CLI launch, opt in explicitly:

Example launches for Qwen3.5-122B-A10B and the built-in `mllm_demo` dataset:

```bash
LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 \
  llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_text_lora_sft_kt.yaml
LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 \
  llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_vision_lora_sft_kt.yaml
LLAMAFACTORY_ALLOW_TRANSFORMERS_KT=1 \
  llamafactory-cli train examples/ktransformers/train_lora/qwen3_5moe_vlm_all_lora_sft_kt.yaml
```

Qwen3-VL-MoE uses the same scoped LoRA path. The minimal all-modality example
is `qwen3vlmoe_vlm_all_lora_sft_kt.yaml`; its matching eight-process FSDP2
configuration is `../accelerate/fsdp2_kt_bf16_qwen3_vl_moe.yaml`. KT expert
offload applies only to the MoE variants such as Qwen3-VL-30B-A3B and
Qwen3-VL-235B-A22B. Run dense Qwen3-VL variants with `use_kt: false`.

On PyTorch 2.9.x, install a KT kernel build with the optional compatibility dependency before launching:

```bash
pip install 'kt-kernel[vlm-sft]'
```
