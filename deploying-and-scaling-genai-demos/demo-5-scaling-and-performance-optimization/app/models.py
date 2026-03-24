from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1024, examples=["Once upon a time"])
    max_tokens: int = Field(default=50, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Prompt must contain non-whitespace characters.")
        return v


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    tokens_generated: int
    model_name: str
    cache_hit: bool = Field(default=False, description="Whether the response was served from cache.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    cache_connected: bool = False


class BatchGenerateRequest(BaseModel):
    """Request model for batched generation -- multiple prompts in one call."""
    prompts: list[str] = Field(..., min_length=1, max_length=8, description="List of prompts (max 8).")
    max_tokens: int = Field(default=50, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BatchGenerateResponse(BaseModel):
    results: list[GenerateResponse]
    batch_size: int
