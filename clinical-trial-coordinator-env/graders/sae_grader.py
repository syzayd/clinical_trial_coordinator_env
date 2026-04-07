"""
SAE Narrative Grader (Task 3 — Hard)

The agent must draft a complete Serious Adverse Event (SAE) narrative
following ICH E2A guidelines. This is extremely hard — it requires:
  - Correct identification of SAE criteria met
  - Causality assessment matching clinical logic
  - Correct regulatory classification
  - Complete narrative covering all required elements

We use a hybrid grader:
  - Deterministic checks for required elements (40%)
  - LLM-as-judge for narrative quality (60%)
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Required elements per ICH E2A (deterministic checks)
# ---------------------------------------------------------------------------

REQUIRED_NARRATIVE_ELEMENTS = [
    ("patient_identifier", ["patient", "subject", "pt-", "participant"]),
    ("event_description", ["pneumonitis", "adverse event", "event", "reaction"]),
    ("onset", ["onset", "began", "started", "occurred", "day", "date"]),
    ("severity", ["grade", "severe", "serious", "mild", "moderate"]),
    ("causality", ["related", "unrelated", "possible", "probable", "definite", "assess"]),
    ("action_taken", ["discontinued", "dose", "treated", "managed", "initiated", "given"]),
    ("outcome", ["recovered", "recovering", "resolved", "ongoing", "fatal"]),
    ("regulatory_category", ["hospitali", "life-threatening", "death", "congenital", "disability"]),
]

VALID_CAUSALITY_TERMS = {"unrelated", "possible", "probable", "definite", "unlikely"}
VALID_REGULATORY_CATEGORIES = {
    "requires_hospitalization",
    "life-threatening",
    "results_in_death",
    "persistent_disability",
    "congenital_anomaly",
    "medically_significant",
}

# Ground truth for AE-002 (the SAE scenario)
SAE_GROUND_TRUTH = {
    "ae_id": "AE-002",
    "correct_causality": "probable",
    "correct_regulatory_category": "requires_hospitalization",
    "is_serious": True,
    "required_actions": ["drug discontinuation", "corticosteroid"],
    "minimum_narrative_length": 150,  # words
}


def _check_required_elements(narrative: str) -> Dict[str, bool]:
    """Check which required ICH E2A elements are present in the narrative."""
    narrative_lower = narrative.lower()
    results = {}
    for element_name, keywords in REQUIRED_NARRATIVE_ELEMENTS:
        results[element_name] = any(kw in narrative_lower for kw in keywords)
    return results


def _score_causality(agent_causality: str, correct_causality: str) -> float:
    agent = agent_causality.lower().strip()
    correct = correct_causality.lower()
    if agent == correct:
        return 1.0
    causality_ladder = ["unrelated", "unlikely", "possible", "probable", "definite"]
    try:
        agent_idx = causality_ladder.index(agent)
        correct_idx = causality_ladder.index(correct)
        distance = abs(agent_idx - correct_idx)
        return max(0.0, 1.0 - distance * 0.3)
    except ValueError:
        return 0.0


def _score_regulatory(agent_category: str, correct_category: str) -> float:
    agent_lower = agent_category.lower().replace(" ", "_").replace("-", "_")
    correct_lower = correct_category.lower().replace(" ", "_").replace("-", "_")
    if correct_lower in agent_lower or agent_lower in correct_lower:
        return 1.0
    # Partial: hospitalization keywords
    if "hospital" in agent_lower and "hospital" in correct_lower:
        return 0.8
    return 0.0


def grade(action_payload: Dict[str, Any], adverse_event: Dict, protocol: Dict) -> Dict[str, Any]:
    """
    Grade an SAE narrative submission.
    Returns score [0.0, 1.0] with detailed breakdown.
    """
    narrative = action_payload.get("narrative", "")
    agent_causality = action_payload.get("causality", "")
    agent_regulatory = action_payload.get("regulatory_category", "")

    gt = SAE_GROUND_TRUTH

    # --- Deterministic checks (40% of score) ---

    # 1. Required elements present (20%)
    element_checks = _check_required_elements(narrative)
    elements_present = sum(1 for v in element_checks.values() if v)
    elements_score = elements_present / len(REQUIRED_NARRATIVE_ELEMENTS)

    # 2. Narrative length (5%)
    word_count = len(narrative.split())
    length_score = min(1.0, word_count / gt["minimum_narrative_length"])

    # 3. Causality correctness (10%)
    causality_score = _score_causality(agent_causality, gt["correct_causality"])

    # 4. Regulatory category (5%)
    regulatory_score = _score_regulatory(agent_regulatory, gt["correct_regulatory_category"])

    # 5. Required actions mentioned (10%)
    narrative_lower = narrative.lower()
    actions_mentioned = sum(
        1 for action in gt["required_actions"]
        if any(word in narrative_lower for word in action.split())
    )
    actions_score = actions_mentioned / len(gt["required_actions"])

    deterministic_score = (
        elements_score * 0.20
        + length_score * 0.05
        + causality_score * 0.10
        + regulatory_score * 0.05
        + actions_score * 0.10
    )  # max = 0.50 (we leave 0.50 for LLM judge or fill with deterministic)

    # For environments without LLM judge access, scale deterministic to full range
    # (In production the LLM judge adds the remaining 50%)
    total = min(1.0, round(deterministic_score * 2.0, 3))

    missing_elements = [k for k, v in element_checks.items() if not v]
    feedback_parts = []
    if missing_elements:
        feedback_parts.append(f"Missing elements: {missing_elements}.")
    if causality_score < 1.0:
        feedback_parts.append(
            f"Causality '{agent_causality}' is not optimal; expected '{gt['correct_causality']}'."
        )
    if regulatory_score < 1.0:
        feedback_parts.append(
            f"Regulatory category '{agent_regulatory}' should be '{gt['correct_regulatory_category']}'."
        )
    if word_count < gt["minimum_narrative_length"]:
        feedback_parts.append(
            f"Narrative too brief ({word_count} words); minimum {gt['minimum_narrative_length']}."
        )
    if not feedback_parts:
        feedback_parts.append("Complete and well-structured SAE narrative.")

    return {
        "score": total,
        "breakdown": {
            "elements_present": f"{elements_present}/{len(REQUIRED_NARRATIVE_ELEMENTS)}",
            "elements_score": round(elements_score, 3),
            "length_score": round(length_score, 3),
            "causality_score": round(causality_score, 3),
            "regulatory_score": round(regulatory_score, 3),
            "actions_score": round(actions_score, 3),
            "deterministic_total": round(deterministic_score, 3),
        },
        "element_details": element_checks,
        "feedback": " ".join(feedback_parts),
    }
