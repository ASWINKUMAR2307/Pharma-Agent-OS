from typing import List, Optional
from pydantic import BaseModel, Field

class CodedEvent(BaseModel):
    verbatim_reported: str
    meddra_pt: str
    meddra_code: str

class SafetyCasePayload(BaseModel):
    patient_age: int
    patient_gender: str
    suspect_product: str
    dosage: str
    coded_adverse_events: List[CodedEvent]
    seriousness_criteria: str
    reporter: str
    outcome: str

class DuplicateResult(BaseModel):
    is_duplicate: bool
    similarity_score: float
    closest_match_id: Optional[str] = None
    match_reason: str

class QCResult(BaseModel):
    passed: bool
    flags: List[str] = []

class RiskAssessment(BaseModel):
    risk_level: str = Field(description="High, Medium, or Low")
    requires_human_review: bool = Field(description="True if a human must look at it")
    routing_reason: str = Field(description="Explanation of why it needs review or can be auto-approved")