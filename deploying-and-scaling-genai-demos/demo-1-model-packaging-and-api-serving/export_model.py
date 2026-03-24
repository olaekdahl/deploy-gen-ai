"""
ONNX Model Export Script

Exports the distilgpt2 model to ONNX format using Hugging Face Optimum.

This demonstrates model serialization concepts covered in the course:
  - Converting a PyTorch model to a portable, framework-agnostic format
  - Benefits of ONNX: cross-platform inference, runtime optimization, graph fusion
  - Trade-offs: export complexity, dynamic shapes, operator coverage

Usage:
    pip install optimum[onnxruntime]
    python export_model.py

After export, start the server with ONNX by setting environment variables:
    USE_ONNX=true ONNX_MODEL_PATH=./onnx_model uvicorn app.main:app --host 0.0.0.0
"""

import os

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

MODEL_NAME = "distilgpt2"
OUTPUT_DIR = "./onnx_model"


def export() -> None:
    print(f"Exporting '{MODEL_NAME}' to ONNX format...")

    # Optimum handles the PyTorch -> ONNX conversion automatically
    model = ORTModelForCausalLM.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Save both model and tokenizer so they can be loaded together later
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel exported to {OUTPUT_DIR}/")
    print("Files created:")
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename:40s} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    export()
