"""
ONNX Runtime inference module.
Loads the exported ONNX model and provides inference functions.
"""

import os
import numpy as np
import onnxruntime as ort
from typing import List, Tuple

# ============== Configuration ==============
ARTIFACTS_DIR = "./artifacts"
ONNX_PATH = os.path.join(ARTIFACTS_DIR, "model.onnx")
CLASS_NAMES = ["class_0", "class_1"]


class ONNXInferenceEngine:
    """Loads ONNX model and runs inference using ONNX Runtime."""
    
    def __init__(self, model_path: str = ONNX_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_path = model_path
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
    def predict(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run inference on input array.
        
        Args:
            inputs: numpy array of shape (batch_size, 2) with float32 dtype
            
        Returns:
            logits: raw model outputs (batch_size, num_classes)
            class_indices: predicted class indices (batch_size,)
            class_names: predicted class names (batch_size,)
        """
        # Ensure correct dtype
        if inputs.dtype != np.float32:
            inputs = inputs.astype(np.float32)
        
        # Run inference
        logits = self.session.run(None, {self.input_name: inputs})[0]
        
        # Post-process: get class predictions
        class_indices = np.argmax(logits, axis=1)
        class_names = [CLASS_NAMES[idx] for idx in class_indices]
        
        return logits, class_indices, class_names
    
    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_path": self.model_path,
            "input_name": self.input_name,
            "input_shape": str(self.input_shape),
            "output_name": self.output_name,
            "providers": self.session.get_providers(),
        }


def main():
    """Demo: run inference on sample inputs."""
    print("=" * 50)
    print("ONNX Runtime Inference Demo")
    print("=" * 50)
    
    # Load model
    engine = ONNXInferenceEngine()
    print(f"\nModel info: {engine.get_model_info()}\n")
    
    # Sample inputs
    test_inputs = np.array([
        [-1.5, -1.5],  # Should be class 0
        [1.5, 1.5],    # Should be class 1
        [-0.5, 0.5],   # Near boundary
        [0.0, 0.0],    # Origin
    ], dtype=np.float32)
    
    # Run inference
    logits, class_indices, class_names = engine.predict(test_inputs)
    
    print("Inference Results:")
    print("-" * 50)
    for i, (inp, logit, cls_idx, cls_name) in enumerate(zip(test_inputs, logits, class_indices, class_names)):
        print(f"  Input {i}: {inp} → logits: {logit} → {cls_name} (idx={cls_idx})")
    print("-" * 50)


if __name__ == "__main__":
    main()
