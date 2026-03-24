"""
Debug script: inspect model internals in PyTorch mode.

This script demonstrates debugging capabilities that are ONLY available
when running in PyTorch mode. In ONNX mode, the model is a static
compiled graph -- you cannot access individual layers, hook into
intermediate outputs, or inspect weight tensors directly.

Usage (PyTorch mode only -- do NOT set USE_ONNX):
    python debug_pytorch.py

Instructor note:
    Run this side-by-side with the ONNX exported model to show students
    what you lose when you compile a model to a static graph.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"

print("=" * 70)
print("  PyTorch Debugging Demo -- Inspecting Model Internals")
print("=" * 70)

# Load the model as a live PyTorch module (not via pipeline, for full access)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

# -------------------------------------------------------------------------
# 1. Inspect model architecture -- list every layer and parameter count
# -------------------------------------------------------------------------
print("\n--- 1. Model Architecture ---")
total_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    # Show the first few layers
print(f"Total parameters: {total_params:,}")
print(f"Total layers: {len(list(model.named_parameters()))}")
print("\nFirst 5 named parameter tensors:")
for i, (name, param) in enumerate(model.named_parameters()):
    if i >= 5:
        break
    print(f"  {name:50s} shape={list(param.shape)}")

# -------------------------------------------------------------------------
# 2. Read a specific weight tensor -- impossible in ONNX
# -------------------------------------------------------------------------
print("\n--- 2. Inspect a Specific Weight Tensor ---")
# Access the first transformer block's attention weights directly
attn_weight = model.transformer.h[0].attn.c_attn.weight
print(f"Layer: transformer.h[0].attn.c_attn.weight")
print(f"  Shape:  {list(attn_weight.shape)}")
print(f"  Dtype:  {attn_weight.dtype}")
print(f"  Mean:   {attn_weight.mean().item():.6f}")
print(f"  Std:    {attn_weight.std().item():.6f}")
print(f"  Min:    {attn_weight.min().item():.6f}")
print(f"  Max:    {attn_weight.max().item():.6f}")

# -------------------------------------------------------------------------
# 3. Register a forward hook to capture intermediate activations
# -------------------------------------------------------------------------
print("\n--- 3. Capture Intermediate Activations via Hook ---")

captured_activations = {}

def hook_fn(layer_name):
    """Factory: returns a hook that stores the layer's output."""
    def hook(module, input, output):
        # output can be a tuple; grab the hidden states
        if isinstance(output, tuple):
            captured_activations[layer_name] = output[0].detach()
        else:
            captured_activations[layer_name] = output.detach()
    return hook

# Attach hooks to the first and last transformer blocks
model.transformer.h[0].register_forward_hook(hook_fn("block_0"))
model.transformer.h[-1].register_forward_hook(hook_fn("block_last"))

# Run a forward pass
prompt = "Debugging deep learning models is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Inspect what the hooks captured
for layer_name, activation in captured_activations.items():
    print(f"\n  Layer: {layer_name}")
    print(f"    Shape: {list(activation.shape)}")
    print(f"    Mean:  {activation.mean().item():.6f}")
    print(f"    Std:   {activation.std().item():.6f}")
    print(f"    [Sample values from position 0, first 5 dims]:")
    print(f"    {activation[0, 0, :5].tolist()}")

# -------------------------------------------------------------------------
# 4. Inspect the output logits and token probabilities
# -------------------------------------------------------------------------
print("\n--- 4. Output Logits Analysis ---")
logits = outputs.logits  # shape: [batch, seq_len, vocab_size]
print(f"Logits shape: {list(logits.shape)}")
print(f"Vocab size:   {logits.shape[-1]}")

# Get the top-5 predicted next tokens
last_token_logits = logits[0, -1, :]
probs = torch.softmax(last_token_logits, dim=-1)
top5 = torch.topk(probs, k=5)

print(f"\nTop 5 next-token predictions for prompt: '{prompt}'")
for i in range(5):
    token_id = top5.indices[i].item()
    token_str = tokenizer.decode([token_id])
    probability = top5.values[i].item()
    print(f"  {i+1}. '{token_str}' (id={token_id}, prob={probability:.4f})")

# -------------------------------------------------------------------------
# 5. Gradient inspection (training-specific, impossible in ONNX)
# -------------------------------------------------------------------------
print("\n--- 5. Gradient Flow Check ---")
model.zero_grad()
# Re-run with gradients enabled
inputs2 = tokenizer("Test gradient flow", return_tensors="pt")
outputs2 = model(**inputs2, labels=inputs2["input_ids"])
loss = outputs2.loss
loss.backward()

print(f"Loss value: {loss.item():.4f}")
grad_norms = []
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norms.append((name, param.grad.norm().item()))

# Show the 5 layers with the largest gradient norms
grad_norms.sort(key=lambda x: x[1], reverse=True)
print("Top 5 layers by gradient magnitude:")
for name, norm in grad_norms[:5]:
    print(f"  {name:50s} grad_norm={norm:.6f}")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  NONE of the above is possible with the ONNX model because:")
print("  - ONNX is a static, pre-compiled graph (no Python layer objects)")
print("  - You cannot register hooks on ONNX Runtime nodes")
print("  - You cannot access individual weight tensors by name")
print("  - You cannot compute gradients (inference-only runtime)")
print("  - You cannot inspect intermediate activations without re-exporting")
print("=" * 70)
