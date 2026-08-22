import torch
from transformers import AutoModelForCausalLM
from llm_edge_compression.compressors import TensorNetworkCompressor

# Load original model
original = AutoModelForCausalLM.from_pretrained("gpt2")
original.eval()

# Make a copy
compressed = AutoModelForCausalLM.from_pretrained("gpt2")
compressed.eval()

# Compress
compressor = TensorNetworkCompressor(rank_ratio=0.5)
compressed = compressor.compress(compressed)

# Compare selected layers
for name, module in original.named_modules():

    if "transformer.h.0" not in name:
        continue

    compressed_module = dict(compressed.named_modules()).get(name)

    if compressed_module is None:
        continue

    if hasattr(module, "weight") and hasattr(compressed_module, "left_core"):
        print("\nLayer:", name)

        weight = module.weight.detach().cpu()

        # GPT-2 Conv1D stores [in, out]
        if weight.shape[0] != weight.shape[1]:
            weight = weight.t()

        in_features = weight.shape[1]
        out_features = weight.shape[0]

        print("Original weight:", weight.shape)
        print("Compressed factors:")
        print("  in:", compressed_module.in_factors)
        print("  out:", compressed_module.out_factors)

        # Reconstruct compressed weight by feeding identity
        identity = torch.eye(in_features)

        with torch.no_grad():
            compressed_output = compressed_module(identity)

        reconstructed = compressed_output.T

        error = torch.norm(weight - reconstructed)
        relative_error = error / torch.norm(weight)

        print("Absolute error:", error.item())
        print("Relative error:", relative_error.item())
