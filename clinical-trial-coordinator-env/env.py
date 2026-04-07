"""
ClinicalTrialCoordinatorEnv — core environment logic.

State machine managing:
  - Task routing (easy/medium/hard)
  - Episode lifecycle (reset → step → done)
  - Reward shaping with partial progress signals
  - Audit logging for reproducibility
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from models import (
    ClinicalTrialAction,
    ClinicalTrialObservation,
    ClinicalTrialReward,
    PatientRecord,
    TrialProtocol,
    VisitRecord,
    AdverseEvent,
)

DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str) -> Any:
    with open(DATA_DIR / filename) as f:
        return json.load(f)


class ClinicalTrialCoordinatorEnv:
    """
    Simulates the full workflow of a clinical research coordinator.

    Task 1 (easy):   Patient eligibility screening
    Task 2 (medium): Protocol deviation detection
    Task 3 (hard):   SAE narrative drafting + regulatory assessment

    API:
        reset(task_name) → ClinicalTrialObservation
        step(action)     → (observation, reward, done, info)
        state()          → current raw state dict
    """

    MAX_STEPS = {
        "screen_patient": 4,
        "detect_deviation": 6,
        "draft_sae_narrative": 8,
    }

    TASK_SCENARIOS = {
        "screen_patient": [
            {"patient_id": "PT-001", "trial_id": "TRIAL-001"},  # eligible
            {"patient_id": "PT-002", "trial_id": "TRIAL-001"},  # ineligible (NYHA III)
            {"patient_id": "PT-003", "trial_id": "TRIAL-001"},  # ineligible (GLP-1)
            {"patient_id": "PT-004", "trial_id": "TRIAL-002"},  # eligible
            {"patient_id": "PT-005", "trial_id": "TRIAL-002"},  # ineligible (autoimmune)
        ],
        "detect_deviation": [
            {"visit_id": "VIS-001", "trial_id": "TRIAL-001"},
            {"visit_id": "VIS-002", "trial_id": "TRIAL-002"},
        ],
        "draft_sae_narrative": [
            {"ae_id": "AE-002", "trial_id": "TRIAL-002"},
        ],
    }

    def __init__(self, task_name: str = "screen_patient", scenario_index: int = 0):
        self.task_name = task_name
        self.scenario_index = scenario_index
        self._step = 0
        self._done = False
        self._audit_log: List[str] = []
        self._score_accumulator = 0.0
        self._last_feedback = ""

        # Load data
        patients_raw = _load_json("patients.json")
        protocols_raw = _load_json("protocols.json")
        ae_data = _load_json("adverse_events.json")

        self._patients: Dict[str, PatientRecord] = {
            p["patient_id"]: PatientRecord(**p) for p in patients_raw
        }
        self._protocols: Dict[str, TrialProtocol] = {
            p["trial_id"]: TrialProtocol(**p) for p in protocols_raw
        }
        self._visits: Dict[str, VisitRecord] = {
            v["visit_id"]: VisitRecord(**v)
            for v in ae_data.get("visit_records", [])
        }
        self._aes: Dict[str, AdverseEvent] = {
            ae["ae_id"]: AdverseEvent(**ae)
            for ae in ae_data.get("adverse_events", [])
        }

        # Resolve current scenario
        scenarios = self.TASK_SCENARIOS.get(task_name, [])
        if not scenarios:
            raise ValueError(f"Unknown task: {task_name}")
        scenario = scenarios[scenario_index % len(scenarios)]

        self._patient: Optional[PatientRecord] = self._patients.get(scenario.get("patient_id"))
        self._protocol: TrialProtocol = self._protocols[scenario["trial_id"]]
        self._visit: Optional[VisitRecord] = self._visits.get(scenario.get("visit_id"))
        self._ae: Optional[AdverseEvent] = self._aes.get(scenario.get("ae_id"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> ClinicalTrialObservation:
        """Reset environment to initial state and return first observation."""
        self._step = 0
        self._done = False
        self._audit_log = ["[RESET] Environment initialized."]
        self._score_accumulator = 0.0
        self._last_feedback = "Study the patient record and protocol carefully, then act."

        return self._build_observation()

    def step(
        self, action: ClinicalTrialAction
    ) -> Tuple[ClinicalTrialObservation, ClinicalTrialReward, bool, Dict]:
        """
        Execute one action and return (observation, reward, done, info).
        Reward is always in [0.0, 1.0].
        """
        if self._done:
            obs = self._build_observation()
            reward = ClinicalTrialReward(
                total=0.0, correctness=0.0, completeness=0.0,
                reasoning_quality=0.0, efficiency_bonus=0.0, penalty=0.0,
                explanation="Episode already complete."
            )
            return obs, reward, True, {"warning": "Episode already done."}

        self._step += 1
        self._audit_log.append(
            f"[STEP {self._step}] action_type={action.action_type}"
        )

        # --- Route to task-specific grader ---
        reward, done, info = self._dispatch(action)

        # Efficiency bonus: reward completing correctly with fewer steps
        if done and reward.total > 0.5:
            max_steps = self.MAX_STEPS.get(self.task_name, 8)
            efficiency = max(0.0, (max_steps - self._step) / max_steps)
            bonus = round(efficiency * 0.10, 3)
            reward.efficiency_bonus = bonus
            reward.total = min(1.0, round(reward.total + bonus, 3))
            reward.explanation += f" Efficiency bonus: +{bonus}."

        # Step budget exhaustion
        max_steps = self.MAX_STEPS.get(self.task_name, 8)
        if self._step >= max_steps and not done:
            done = True
            penalty = 0.15
            reward.penalty = penalty
            reward.total = max(0.0, round(reward.total - penalty, 3))
            reward.explanation += f" Step budget exhausted (penalty -{penalty})."
            self._audit_log.append("[DONE] Step budget exhausted.")

        self._done = done
        self._last_feedback = reward.explanation
        self._score_accumulator += reward.total

        obs = self._build_observation()
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return raw state dict for debugging and spec compliance."""
        return {
            "task_name": self.task_name,
            "scenario_index": self.scenario_index,
            "step": self._step,
            "done": self._done,
            "score_accumulator": self._score_accumulator,
            "audit_log": self._audit_log,
            "patient_id": self._patient.patient_id if self._patient else None,
            "trial_id": self._protocol.trial_id,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> ClinicalTrialObservation:
        max_steps = self.MAX_STEPS.get(self.task_name, 8)
        return ClinicalTrialObservation(
            task_name=self.task_name,
            step=self._step,
            patient=self._patient or PatientRecord(
                patient_id="N/A", age=0, sex="N/A", diagnosis="N/A",
                comorbidities=[], current_medications=[], lab_values={},
                visit_dates=[], adverse_events=[]
            ),
            protocol=self._protocol,
            current_visit=self._visit,
            adverse_event=self._ae,
            audit_log=self._audit_log[-10:],  # last 10 entries
            steps_remaining=max_steps - self._step,
            feedback=self._last_feedback,
            score_so_far=round(self._score_accumulator, 3),
        )

    def _dispatch(
        self, action: ClinicalTrialAction
    ) -> Tuple[ClinicalTrialReward, bool, Dict]:
        """Route action to the correct grader and return (reward, done, info)."""

        if action.action_type == "request_info":
            # Agent asks a clarifying question — costs a step, no reward
            self._audit_log.append(f"  [INFO_REQUEST] {action.payload.get('question', '')}")
            reward = ClinicalTrialReward(
                total=0.0, correctness=0.0, completeness=0.0,
                reasoning_quality=0.0, efficiency_bonus=0.0, penalty=0.0,
                explanation="Info request noted. Use your remaining steps to act."
            )
            return reward, False, {"info_request": True}

        if action.action_type == "finalize":
            reward = ClinicalTrialReward(
                total=0.0, correctness=0.0, completeness=0.0,
                reasoning_quality=0.0, efficiency_bonus=0.0, penalty=0.05,
                explanation="Episode finalized without completing the task."
            )
            return reward, True, {"finalized_early": True}

        # ---- Task 1: Eligibility screening ----
        if action.action_type == "screen_patient" and self.task_name == "screen_patient":
            from graders.eligibility_grader import grade
            result = grade(
                action.payload,
                self._patient.model_dump() if self._patient else {},
                self._protocol.model_dump(),
            )
            done = True  # single-step terminal task
            reward = _build_reward_from_grader(result)
            self._audit_log.append(f"  [GRADE] score={result['score']:.3f}")
            return reward, done, result

        # ---- Task 2: Protocol deviation detection ----
        if action.action_type == "flag_deviation" and self.task_name == "detect_deviation":
            from graders.deviation_grader import grade
            result = grade(
                action.payload,
                self._visit.model_dump() if self._visit else {},
                self._protocol.model_dump(),
            )
            done = True
            reward = _build_reward_from_grader(result)
            self._audit_log.append(f"  [GRADE] score={result['score']:.3f}")
            return reward, done, result

        # ---- Task 3: SAE narrative ----
        if action.action_type == "draft_sae_narrative" and self.task_name == "draft_sae_narrative":
            from graders.sae_grader import grade
            result = grade(
                action.payload,
                self._ae.model_dump() if self._ae else {},
                self._protocol.model_dump(),
            )
            done = True
            reward = _build_reward_from_grader(result)
            self._audit_log.append(f"  [GRADE] score={result['score']:.3f}")
            return reward, done, result

        # Unrecognized action for this task
        reward = ClinicalTrialReward(
            total=0.0, correctness=0.0, completeness=0.0,
            reasoning_quality=0.0, efficiency_bonus=0.0, penalty=0.05,
            explanation=f"Action '{action.action_type}' not valid for task '{self.task_name}'. "
                        f"Use the correct action type."
        )
        return reward, False, {"error": "invalid_action_type"}


def _build_reward_from_grader(result: Dict) -> ClinicalTrialReward:
    breakdown = result.get("breakdown", {})
    return ClinicalTrialReward(
        total=result["score"],
        correctness=breakdown.get("decision_correct", breakdown.get("detection", result["score"])),
        completeness=breakdown.get("completeness", result["score"]),
        reasoning_quality=breakdown.get("reasoning", 0.0),
        efficiency_bonus=0.0,
        penalty=0.0,
        explanation=result.get("feedback", ""),
    )
