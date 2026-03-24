"""Pydantic request and response models for the GenAI API."""

from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="The input text prompt for generation.",
        examples=["Once upon a time in a land far away"],
    )
    max_tokens: int = Field(
        default=50, ge=1, le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0,
        description="Sampling temperature.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Prompt must contain non-whitespace characters.")
        return v


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
