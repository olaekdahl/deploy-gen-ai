"""
Pydantic request and response models for the GenAI text generation API.

Instructor note:
  Pydantic models serve as both validation and documentation. FastAPI uses
  these models to auto-generate the OpenAPI schema, validate incoming JSON,
  and serialize outgoing responses. This keeps the API contract explicit.
"""

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="The input text prompt for generation.",
        examples=["Once upon a time in a land far away"],
    )
    max_tokens: int = Field(
        default=50,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values increase randomness.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v: str) -> str:
        """Reject prompts that are whitespace-only."""
        if not v.strip():
            raise ValueError("Prompt must contain non-whitespace characters.")
        return v


class GenerateResponse(BaseModel):
    """Response body returned by the /generate endpoint."""

    prompt: str = Field(description="The original input prompt.")
    generated_text: str = Field(description="The generated text continuation.")
    tokens_generated: int = Field(description="Number of new tokens produced.")
    model_name: str = Field(description="Identifier of the model used.")


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    model_loaded: bool
    model_name: str
