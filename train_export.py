"""
Train a tiny MLP classifier on synthetic 2D data and export to ONNX.
Validates export by comparing PyTorch vs ONNX Runtime outputs.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import onnx
import onnxruntime as ort

# ============== Configuration ==============
SEED = 42
N_SAMPLES = 200
INPUT_DIM = 2
HIDDEN_DIM = 16
NUM_CLASSES = 2
EPOCHS = 100
LR = 0.1
ARTIFACTS_DIR = "./artifacts"
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")
OPSET_VERSION = 17  # Stable opset with broad compatibility


class TinyMLP(nn.Module):
    """A minimal 2-layer MLP for binary classification."""
    
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits


def create_synthetic_data(n_samples=N_SAMPLES, seed=SEED):
    """
    Create two 2D blobs (clusters) for binary classification.
    Class 0: centered at (-1, -1)
    Class 1: centered at (+1, +1)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    n_per_class = n_samples // 2
    
    # Class 0: blob at (-1, -1)
    X0 = np.random.randn(n_per_class, 2) * 0.5 + np.array([-1, -1])
    y0 = np.zeros(n_per_class, dtype=np.int64)
    
    # Class 1: blob at (+1, +1)
    X1 = np.random.randn(n_per_class, 2) * 0.5 + np.array([1, 1])
    y1 = np.ones(n_per_class, dtype=np.int64)
    
    # Combine and shuffle
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_model(model, X, y, epochs=EPOCHS, lr=LR):
    """Train the model with CrossEntropy loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    print(f"Training for {epochs} epochs...")
    print("-" * 40)
    
    for epoch in range(epochs):
        # Forward
        logits = model(X)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == y).float().mean().item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1:3d}/{epochs}] | Loss: {loss.item():.4f} | Acc: {accuracy:.2%}")
    
    print("-" * 40)
    final_acc = (model(X).argmax(dim=1) == y).float().mean().item()
    print(f"Training complete! Final accuracy: {final_acc:.2%}")
    return model


def export_to_onnx(model, onnx_path=ONNX_PATH):
    """Export PyTorch model to ONNX format."""
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    model.eval()
    
    # Dummy input for tracing (batch of 1, 2D features)
    dummy_input = torch.randn(1, INPUT_DIM)
    
    print(f"\nExporting to ONNX (opset {OPSET_VERSION})...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},   # Variable batch size
            "output": {0: "batch_size"}
        }
    )
    
    # Validate ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model saved and validated: {onnx_path}")
    
    return onnx_path


def validate_export(model, onnx_path=ONNX_PATH):
    """Compare PyTorch and ONNX Runtime outputs."""
    print("\nValidating ONNX export...")
    
    model.eval()
    
    # Test samples
    test_inputs = torch.tensor([
        [-1.0, -1.0],  # Should be class 0
        [1.0, 1.0],    # Should be class 1
        [0.0, 0.0],    # Edge case
    ], dtype=torch.float32)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_inputs).numpy()
    
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"input": test_inputs.numpy()})[0]
    
    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    
    print(f"\nTest inputs:\n{test_inputs.numpy()}")
    print(f"\nPyTorch output:\n{pytorch_output}")
    print(f"\nONNX Runtime output:\n{onnx_output}")
    print(f"\nMax absolute difference: {max_diff:.2e}")
    
    if max_diff < 1e-5:
        print("✓ Validation passed: outputs match!")
    else:
        print("⚠ Warning: outputs differ more than expected")
    
    return max_diff


def main():
    print("=" * 50)
    print("PyTorch → ONNX Training & Export Pipeline")
    print(f"PyTorch: {torch.__version__} | ONNX Runtime: {ort.__version__}")
    print("=" * 50)
    
    # Create data
    X, y = create_synthetic_data()
    print(f"\nDataset: {len(X)} samples, {INPUT_DIM}D features, {NUM_CLASSES} classes")
    print(f"Class distribution: {(y == 0).sum().item()} class-0, {(y == 1).sum().item()} class-1\n")
    
    # Create and train model
    model = TinyMLP()
    print(f"Model:\n{model}\n")
    
    trained_model = train_model(model, X, y)
    
    # Export to ONNX
    export_to_onnx(trained_model)
    
    # Validate
    validate_export(trained_model)
    
    print("\n" + "=" * 50)
    print("Pipeline complete! Model ready for inference.")
    print(f"ONNX model: {ONNX_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
