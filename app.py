"""
FastAPI service for ONNX model inference.
Provides /health and /predict endpoints.
"""

import numpy as np
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from inference_onnx import ONNXInferenceEngine

# ============== Configuration ==============
MODEL_PATH = "./artifacts/model.onnx"
EXPECTED_FEATURES = 2


# ============== Pydantic Schemas ==============
class PredictRequest(BaseModel):
    """
    Request body for /predict endpoint.
    
    Example:
        {"inputs": [[0.1, -0.2], [1.0, 2.0]]}
    """
    inputs: List[List[float]] = Field(
        ...,
        description="List of 2D points. Each point is [x, y].",
        json_schema_extra={"example": [[0.1, -0.2], [1.0, 2.0]]}
    )
    
    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v):
        if not v:
            raise ValueError("inputs cannot be empty")
        for i, point in enumerate(v):
            if len(point) != EXPECTED_FEATURES:
                raise ValueError(
                    f"Point at index {i} has {len(point)} features, expected {EXPECTED_FEATURES}"
                )
        return v


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    logits: List[List[float]] = Field(
        description="Raw model outputs (logits) for each input"
    )
    class_indices: List[int] = Field(
        description="Predicted class index for each input"
    )
    class_names: List[str] = Field(
        description="Predicted class name for each input"
    )


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    model_loaded: bool
    model_path: str


# ============== Application ==============
# Global inference engine (loaded at startup)
inference_engine: ONNXInferenceEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, cleanup at shutdown."""
    global inference_engine
    print(f"Loading ONNX model from {MODEL_PATH}...")
    try:
        inference_engine = ONNXInferenceEngine(MODEL_PATH)
        print(f"✓ Model loaded successfully")
        print(f"  Model info: {inference_engine.get_model_info()}")
    except FileNotFoundError as e:
        print(f"✗ Failed to load model: {e}")
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_export.py first.")
    
    yield  # Application runs here
    
    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(
    title="PyTorch ONNX Inference API",
    description="A simple API serving a binary classifier trained on 2D synthetic data.",
    version="1.0.0",
    lifespan=lifespan
)


# ============== Endpoints ==============
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the service is running and model is loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=inference_engine is not None,
        model_path=MODEL_PATH
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(request: PredictRequest):
    """
    Run inference on 2D input points.
    
    **Input format:**
    ```json
    {"inputs": [[x1, y1], [x2, y2], ...]}
    ```
    
    **Example:**
    - Points near (-1, -1) → class_0
    - Points near (+1, +1) → class_1
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to numpy array
        inputs = np.array(request.inputs, dtype=np.float32)
        
        # Run inference
        logits, class_indices, class_names = inference_engine.predict(inputs)
        
        return PredictResponse(
            logits=logits.tolist(),
            class_indices=class_indices.tolist(),
            class_names=class_names
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "PyTorch ONNX Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Run inference",
            "/docs": "GET - Interactive API documentation"
        }
    }
