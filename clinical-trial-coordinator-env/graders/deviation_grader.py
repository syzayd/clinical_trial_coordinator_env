"""
Protocol Deviation Detection Grader (Task 2 — Medium)

The agent reviews a visit record against a trial protocol and must:
1. Identify all protocol deviations
2. Classify each as minor / major / critical
3. Suggest corrective actions

The grader has ground-truth deviation lists per scenario.
"""
from __future__ import annotations
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Ground truth deviation database (deterministic per visit_id)
# ---------------------------------------------------------------------------

GROUND_TRUTH: Dict[str, List[Dict]] = {
    "VIS-001": [
        {
            "type": "visit_timing",
            "description": "Visit occurred on Day 35 instead of scheduled Day 28 (7-day window exceeded for most protocols)",
            "severity": "minor",
            "corrective_action": "Document in deviation log; no protocol amendment required if within allowed window"
        },
        {
            "type": "missing_procedure",
            "description": "ECG not performed as required at Week 4 visit",
            "severity": "major",
            "corrective_action": "Perform ECG at next available opportunity; document reason in source records; notify sponsor"
        }
    ],
    "VIS-002": [
        {
            "type": "missing_procedure",
            "description": "PK sample not collected due to patient refusal",
            "severity": "major",
            "corrective_action": "Document patient refusal in source; notify sponsor; assess impact on PK data completeness"
        }
    ]
}


def grade(action_payload: Dict[str, Any], visit_record: Dict, protocol: Dict) -> Dict[str, Any]:
    """
    Grade a protocol deviation detection action.
    """
    visit_id = visit_record.get("visit_id", "")
    ground_truth = GROUND_TRUTH.get(visit_id, [])

    agent_deviation_type = action_payload.get("deviation_type", "").lower()
    agent_description = action_payload.get("description", "").lower()
    agent_severity = action_payload.get("severity", "").lower()

    if not ground_truth:
        # No deviations exist — if agent flagged one, penalize
        if agent_deviation_type:
            return {
                "score": 0.2,
                "feedback": "No deviations present for this visit; agent incorrectly flagged a deviation.",
                "breakdown": {"detection": 0.0, "severity": 0.0, "completeness": 0.5}
            }
        return {
            "score": 1.0,
            "feedback": "Correct: no deviations present and agent correctly identified none.",
            "breakdown": {"detection": 1.0, "severity": 1.0, "completeness": 1.0}
        }

    # Match agent's flagged deviation to any ground truth deviation
    best_match = None
    best_match_score = 0.0

    for gt in ground_truth:
        # Check if agent mentioned the key concept
        gt_keywords = set(gt["description"].lower().split())
        agent_words = set(agent_description.split())
        overlap = len(gt_keywords & agent_words) / max(1, len(gt_keywords))

        type_match = gt["type"].replace("_", " ") in agent_deviation_type or \
                     any(w in agent_deviation_type for w in gt["type"].split("_"))

        match_score = overlap * 0.7 + (0.3 if type_match else 0.0)
        if match_score > best_match_score:
            best_match_score = match_score
            best_match = gt

    # Detection score: did agent find a real deviation?
    detection_score = min(1.0, best_match_score * 1.5)  # scale up

    # Severity score: did agent classify it correctly?
    severity_score = 0.0
    if best_match:
        correct_severity = best_match["severity"]
        if agent_severity == correct_severity:
            severity_score = 1.0
        elif (agent_severity == "major" and correct_severity == "minor") or \
             (agent_severity == "minor" and correct_severity == "major"):
            severity_score = 0.4  # close but wrong
        elif agent_severity == "critical" and correct_severity in ("major", "critical"):
            severity_score = 0.7  # over-classified but in right direction

    # Completeness: how many total deviations did agent find vs total present?
    # (simplified: we grade one action at a time, so award partial credit)
    completeness_score = min(1.0, detection_score)

    total = round(
        detection_score * 0.50
        + severity_score * 0.30
        + completeness_score * 0.20,
        3
    )

    feedback = []
    if detection_score < 0.5:
        feedback.append(f"Did not correctly identify the deviation. Ground truth: {[g['type'] for g in ground_truth]}.")
    if best_match and severity_score < 1.0:
        feedback.append(f"Severity mismatch: expected '{best_match['severity']}', got '{agent_severity}'.")
    if total >= 0.8:
        feedback.append("Good deviation detection with appropriate severity classification.")

    return {
        "score": total,
        "breakdown": {
            "detection": round(detection_score, 3),
            "severity_classification": round(severity_score, 3),
            "completeness": round(completeness_score, 3),
        },
        "ground_truth_deviations": ground_truth,
        "feedback": " ".join(feedback) or "Deviation detected and classified.",
    }
