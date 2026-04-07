"""
Eligibility Screening Grader (Task 1 — Easy)

The agent receives a patient record and a trial protocol.
It must decide: eligible or ineligible, and provide the exact criteria
that apply. The grader checks decision correctness and reasoning completeness.
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Ground truth logic (deterministic)
# ---------------------------------------------------------------------------

def _evaluate_patient(patient: Dict, protocol: Dict) -> Tuple[str, List[str], List[str]]:
    """
    Returns (decision, violated_exclusion_criteria, met_inclusion_criteria)
    """
    trial_id = protocol["trial_id"]
    lab = patient.get("lab_values", {})
    meds = [m.lower() for m in patient.get("current_medications", [])]
    comorbidities = [c.lower() for c in patient.get("comorbidities", [])]
    age = patient.get("age", 0)

    violated = []
    met_inclusion = []

    if trial_id == "TRIAL-001":
        # --- Inclusion checks ---
        if 30 <= age <= 75:
            met_inclusion.append("Age 30-75 years")
        if "type 2 diabetes mellitus" in patient.get("diagnosis", "").lower():
            met_inclusion.append("Type 2 diabetes diagnosis")
        hba1c = lab.get("HbA1c", 0)
        if 7.5 <= hba1c <= 11.0:
            met_inclusion.append(f"HbA1c in range ({hba1c}%)")
        egfr = lab.get("eGFR", 0)
        if egfr >= 30:
            met_inclusion.append(f"eGFR >= 30 ({egfr})")
        weight = lab.get("body_weight_kg", 0)
        if 50 <= weight <= 130:
            met_inclusion.append(f"Body weight in range ({weight} kg)")

        # --- Exclusion checks ---
        if "heart failure nyha class iii" in comorbidities or "heart failure nyha class iv" in comorbidities:
            violated.append("Heart failure (NYHA Class III or IV)")
        if any("glp-1" in m or "semaglutide" in m or "liraglutide" in m or "dulaglutide" in m for m in meds):
            violated.append("Current use of GLP-1 receptor agonists")
        if "active malignancy" in " ".join(comorbidities):
            violated.append("Active malignancy within last 5 years")
        alt = lab.get("ALT", 0)
        if alt > 120:  # 3x ULN (~40) = 120
            violated.append(f"Liver disease (ALT > 3x ULN): {alt}")

    elif trial_id == "TRIAL-002":
        # --- Inclusion checks ---
        if age >= 18:
            met_inclusion.append("Age >= 18 years")
        ecog = lab.get("ECOG_PS", 99)
        if ecog <= 1:
            met_inclusion.append(f"ECOG PS 0 or 1 ({int(ecog)})")
        anc = lab.get("ANC", 0)
        if anc >= 1.5:
            met_inclusion.append(f"ANC adequate ({anc})")
        platelets = lab.get("platelets", 0)
        if platelets >= 100:
            met_inclusion.append(f"Platelets adequate ({platelets})")

        # --- Exclusion checks ---
        if any("rheumatoid" in c or "autoimmune" in c or "lupus" in c for c in comorbidities):
            violated.append("Active autoimmune disease requiring systemic treatment")
        if any("adalimumab" in m or "methotrexate" in m or "rituximab" in m for m in meds):
            violated.append("Concurrent use of strong immunosuppressants")
        if any("anti-pd" in m or "nivolumab" in m or "pembrolizumab" in m for m in meds):
            violated.append("Prior treatment with anti-PD-1/PD-L1 agents")

    decision = "ineligible" if violated else "eligible"
    return decision, violated, met_inclusion


def grade(action_payload: Dict[str, Any], patient: Dict, protocol: Dict) -> Dict[str, Any]:
    """
    Grade an agent's eligibility screening decision.

    Returns a dict with:
      score: float [0.0, 1.0]
      breakdown: dict with sub-scores
      feedback: str
    """
    correct_decision, correct_violations, correct_inclusions = _evaluate_patient(patient, protocol)

    agent_decision = action_payload.get("decision", "").lower()
    agent_reason = action_payload.get("reason", "")
    agent_violations = [v.lower() for v in action_payload.get("criteria_violated", [])]

    # --- Sub-scores ---

    # 1. Correct yes/no decision (50 pts)
    decision_score = 1.0 if agent_decision == correct_decision else 0.0

    # 2. Violation completeness (30 pts): partial credit for each correct criterion named
    if correct_violations:
        named_correctly = sum(
            1 for cv in correct_violations
            if any(word in " ".join(agent_violations) for word in cv.lower().split()[:3])
        )
        completeness_score = named_correctly / len(correct_violations)
    else:
        # Patient is eligible — agent shouldn't list violations
        completeness_score = 1.0 if not agent_violations else 0.5

    # 3. Reasoning quality (20 pts): checks mention of key lab values / conditions
    reasoning_score = 0.0
    key_terms = []
    if correct_violations:
        for v in correct_violations:
            key_terms.extend(v.lower().split())
    else:
        for inc in correct_inclusions[:3]:
            key_terms.extend(inc.lower().split())
    reason_lower = agent_reason.lower()
    if key_terms:
        hits = sum(1 for t in key_terms if t in reason_lower and len(t) > 3)
        reasoning_score = min(1.0, hits / max(1, len(key_terms) * 0.4))

    # --- Combined score ---
    total = round(
        decision_score * 0.50
        + completeness_score * 0.30
        + reasoning_score * 0.20,
        3
    )

    feedback_parts = []
    if decision_score == 0:
        feedback_parts.append(
            f"Incorrect decision: agent said '{agent_decision}', correct is '{correct_decision}'."
        )
    if correct_violations and completeness_score < 1.0:
        feedback_parts.append(
            f"Missed criteria: {[v for v in correct_violations]}."
        )
    if not feedback_parts:
        feedback_parts.append("Correct and well-reasoned eligibility assessment.")

    return {
        "score": total,
        "breakdown": {
            "decision_correct": decision_score,
            "completeness": completeness_score,
            "reasoning": reasoning_score,
        },
        "correct_decision": correct_decision,
        "correct_violations": correct_violations,
        "feedback": " ".join(feedback_parts),
    }
