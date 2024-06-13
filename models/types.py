from pydantic import BaseModel, Field

class PainService(BaseModel):
    pain_point: str
    service_offering: str


class EvalSet(BaseModel):
    original_text: str
    output: PainService


class PainScore(BaseModel):
    score: float = Field(
        ...,
        description="Score between 0 and 5 of the relevance of pain point and service offering to the original job posting.",
    )


class ServiceScore(BaseModel):
    score: float = Field(
        ...,
        description="Score between 0 and 5 of the relevance of the service offering to the original job posting and the pain point.",
    )