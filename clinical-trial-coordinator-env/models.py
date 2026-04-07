"""
Typed Pydantic models for ClinicalTrialCoordinatorEnv.
Observation, Action, Reward — full OpenEnv spec compliance.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared domain types
# ---------------------------------------------------------------------------

class PatientRecord(BaseModel):
    patient_id: str
    age: int
    sex: str                        # "M" | "F"
    diagnosis: str
    comorbidities: List[str]
    current_medications: List[str]
    lab_values: Dict[str, float]    # e.g. {"eGFR": 45.0, "HbA1c": 8.2}
    visit_dates: List[str]          # ISO date strings
    adverse_events: List[str]


# 🔥 NEW: Proper structured model for safety thresholds
class SafetyThreshold(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    action: Optional[str] = None   # e.g. "dose_reduction", "hold_drug", "alert"


class TrialProtocol(BaseModel):
    trial_id: str
    title: str
    phase: str                      # "I" | "II" | "III"
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    visit_schedule: Dict[str, int]  # {"baseline": 0, "week_4": 28, ...}
    prohibited_medications: List[str]

    # ✅ FIXED: replaced Dict[str, Dict[str, float]]
    safety_thresholds: Dict[str, SafetyThreshold]


class VisitRecord(BaseModel):
    visit_id: str
    patient_id: str
    scheduled_day: int
    actual_day: int
    procedures_done: List[str]
    procedures_required: List[str]
    vitals: Dict[str, float]
    notes: str


class AdverseEvent(BaseModel):
    ae_id: str
    patient_id: str
    event_description: str
    onset_date: str
    severity: str                   # "mild" | "moderate" | "severe" | "life-threatening"
    relatedness: str                # "unrelated" | "possible" | "probable" | "definite"
    outcome: str
    action_taken: str


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class ClinicalTrialObservation(BaseModel):
    task_name: str
    step: int
    patient: PatientRecord
    protocol: TrialProtocol
    current_visit: Optional[VisitRecord] = None
    adverse_event: Optional[AdverseEvent] = None
    audit_log: List[str] = Field(default_factory=list)
    steps_remaining: int
    feedback: str = ""
    score_so_far: float = 0.0


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ClinicalTrialAction(BaseModel):
    action_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class ClinicalTrialReward(BaseModel):
    total: float
    correctness: float
    completeness: float
    reasoning_quality: float
    efficiency_bonus: float
    penalty: float
    explanation: str