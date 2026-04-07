"""
Local smoke test — runs all 3 tasks directly (no HTTP server needed).
Confirms graders work, rewards are in [0.0, 1.0], and state transitions are clean.

Run:
    cd clinical-trial-coordinator-env
    pip install -r requirements.txt
    python test_local.py
"""
import sys
import json
from env import ClinicalTrialCoordinatorEnv
from models import ClinicalTrialAction


PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []


def check(condition: bool, label: str, detail: str = ""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)


# ===========================================================================
# Task 1: screen_patient — ineligible case (PT-002, NYHA III HF)
# ===========================================================================
print("\n" + "="*60)
print("TASK 1: screen_patient — ineligible case (PT-002)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="screen_patient", scenario_index=1)
obs = env.reset()

check(obs.task_name == "screen_patient", "reset() returns correct task_name")
check(obs.step == 0, "reset() starts at step 0")
check(obs.patient.patient_id == "PT-002", f"Correct patient loaded", obs.patient.patient_id)
check(obs.steps_remaining == 4, "Correct step budget")
check(len(obs.audit_log) > 0, "Audit log initialized")

# Wrong action (eligible for ineligible patient) → should score low
action = ClinicalTrialAction(
    action_type="screen_patient",
    payload={
        "decision": "eligible",
        "reason": "Patient meets all criteria.",
        "criteria_violated": [],
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done after terminal action")
check(0.0 <= reward.total <= 1.0, f"Reward in [0,1]: {reward.total}")
check(reward.total < 0.4, f"Low score for wrong decision (got {reward.total:.3f})")
check(reward.correctness == 0.0, "Correctness=0 for wrong decision")
print(f"  INFO  Feedback: {reward.explanation}")

# ===========================================================================
# Task 1: screen_patient — correct ineligible (PT-002, NYHA III HF)
# ===========================================================================
print("\n" + "="*60)
print("TASK 1: screen_patient — correct decision + full criteria")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="screen_patient", scenario_index=1)
env.reset()

action = ClinicalTrialAction(
    action_type="screen_patient",
    payload={
        "decision": "ineligible",
        "reason": "Patient has Heart Failure NYHA Class III, which is an explicit exclusion criterion. "
                  "eGFR is 38 (>30, meets inclusion), but HF NYHA III alone disqualifies the patient.",
        "criteria_violated": ["Heart failure (NYHA Class III or IV)"],
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done")
check(reward.total >= 0.7, f"High score for correct decision ({reward.total:.3f})")
check(reward.correctness == 1.0, "correctness=1 for correct decision")
print(f"  INFO  Score: {reward.total:.3f} | Feedback: {reward.explanation}")

# ===========================================================================
# Task 1: screen_patient — eligible case (PT-001)
# ===========================================================================
print("\n" + "="*60)
print("TASK 1: screen_patient — eligible case (PT-001)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="screen_patient", scenario_index=0)
env.reset()

action = ClinicalTrialAction(
    action_type="screen_patient",
    payload={
        "decision": "eligible",
        "reason": "Patient meets all inclusion criteria: age 58 (in range 30-75), T2DM diagnosis, "
                  "HbA1c 8.4% (in range 7.5-11.0), eGFR 52 (>=30), weight 82kg (in 50-130kg range). "
                  "No exclusion criteria are violated. Not on GLP-1 agonists, no heart failure, "
                  "no active malignancy, ALT 28 (well below 3x ULN).",
        "criteria_violated": [],
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done")
check(reward.total >= 0.7, f"High score for correct eligible decision ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f} | Feedback: {reward.explanation}")

# ===========================================================================
# Task 1: GLP-1 exclusion case (PT-003 on semaglutide)
# ===========================================================================
print("\n" + "="*60)
print("TASK 1: screen_patient — GLP-1 exclusion (PT-003)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="screen_patient", scenario_index=2)
obs = env.reset()
check("PT-003" in obs.patient.patient_id, "PT-003 loaded", obs.patient.patient_id)

action = ClinicalTrialAction(
    action_type="screen_patient",
    payload={
        "decision": "ineligible",
        "reason": "Patient is currently taking Semaglutide 1mg weekly, a GLP-1 receptor agonist, "
                  "which is explicitly listed as an exclusion criterion.",
        "criteria_violated": ["Current use of GLP-1 receptor agonists"],
    }
)
obs2, reward, done, info = env.step(action)
check(reward.total >= 0.7, f"High score for correct GLP-1 exclusion ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f}")

# ===========================================================================
# Task 2: detect_deviation — visit timing + missing ECG
# ===========================================================================
print("\n" + "="*60)
print("TASK 2: detect_deviation — VIS-001 (timing + missing ECG)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="detect_deviation", scenario_index=0)
obs = env.reset()
check(obs.task_name == "detect_deviation", "Correct task name")
check(obs.current_visit is not None, "Visit record loaded")
check(obs.current_visit.visit_id == "VIS-001", "VIS-001 loaded", str(obs.current_visit.visit_id))
check(obs.steps_remaining == 6, "6-step budget for medium task")

# Flag the more serious missing ECG deviation
action = ClinicalTrialAction(
    action_type="flag_deviation",
    payload={
        "deviation_type": "missing_procedure",
        "description": "ECG was not performed at the Week 4 visit despite being a required procedure. "
                       "The visit record confirms ECG was listed as required but not completed due to equipment downtime.",
        "severity": "major",
        "corrective_action": "Perform ECG at next available opportunity. Document reason for omission in source records. Notify sponsor and document in deviation log."
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done")
check(0.0 <= reward.total <= 1.0, f"Reward in [0,1]: {reward.total}")
check(reward.total >= 0.6, f"Score for correct major deviation detection ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f} | Feedback: {reward.explanation}")

# ===========================================================================
# Task 2: detect_deviation — VIS-002 (missing PK sample)
# ===========================================================================
print("\n" + "="*60)
print("TASK 2: detect_deviation — VIS-002 (PK sample refused)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="detect_deviation", scenario_index=1)
obs = env.reset()
check(obs.current_visit.visit_id == "VIS-002", "VIS-002 loaded")

action = ClinicalTrialAction(
    action_type="flag_deviation",
    payload={
        "deviation_type": "missing_procedure",
        "description": "Pharmacokinetic (PK) sample was not collected at Cycle 1 Day 15 visit. "
                       "Patient refused sample collection. This is a required procedure per protocol.",
        "severity": "major",
        "corrective_action": "Document patient refusal in source documents. Notify sponsor immediately. Assess impact on PK data completeness for the study."
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done")
check(reward.total >= 0.55, f"Score for PK deviation detection ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f}")

# ===========================================================================
# Task 2: wrong severity (minor for actually major)
# ===========================================================================
print("\n" + "="*60)
print("TASK 2: detect_deviation — wrong severity classification")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="detect_deviation", scenario_index=0)
env.reset()

action = ClinicalTrialAction(
    action_type="flag_deviation",
    payload={
        "deviation_type": "missing_procedure",
        "description": "ECG was missed at week 4 visit.",
        "severity": "minor",  # wrong — should be major
        "corrective_action": "Document it."
    }
)
obs2, reward, done, info = env.step(action)
check(reward.total < 0.75, f"Penalised for wrong severity ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f} (lower due to severity mismatch)")

# ===========================================================================
# Task 3: draft_sae_narrative — complete narrative
# ===========================================================================
print("\n" + "="*60)
print("TASK 3: draft_sae_narrative — complete narrative")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="draft_sae_narrative", scenario_index=0)
obs = env.reset()
check(obs.task_name == "draft_sae_narrative", "Correct task name")
check(obs.adverse_event is not None, "AE loaded")
check(obs.adverse_event.ae_id == "AE-002", "AE-002 loaded")
check(obs.steps_remaining == 8, "8-step budget for hard task")

full_narrative = (
    "Patient PT-004, a 62-year-old female with confirmed advanced Non-Small Cell Lung Cancer (NSCLC) "
    "enrolled in Study BK-9901, experienced a Grade 3 pneumonitis requiring hospitalization on Day 47 "
    "of study drug treatment (onset date: 2025-03-20). "
    "The patient had received BK-9901 as monotherapy since Day 0 (cycle 1, day 1 = 2025-02-01). "
    "Relevant medical history includes hypothyroidism, managed with levothyroxine 100mcg QD. "
    "No prior immunotherapy or anti-PD-1/PD-L1 agents were administered. "
    "On Day 47, the patient presented with worsening dyspnea and hypoxia (SpO2 82% on room air). "
    "CT chest confirmed bilateral ground-glass opacities consistent with immune-mediated pneumonitis. "
    "Study drug BK-9901 was permanently discontinued immediately upon diagnosis. "
    "Intravenous methylprednisolone 2mg/kg/day was initiated. "
    "The patient was hospitalized for 8 days and showed gradual improvement. "
    "At the time of this report, the patient is recovering with SpO2 94% on 2L nasal cannula. "
    "Causality assessment: The temporal relationship (onset on Day 47 of BK-9901), the known mechanism "
    "of immune checkpoint-related pneumonitis, and absence of alternative causes (no infection identified, "
    "no new medications) support a PROBABLE causal relationship with BK-9901. "
    "This SAE meets the regulatory criterion of requiring hospitalization."
)

action = ClinicalTrialAction(
    action_type="draft_sae_narrative",
    payload={
        "narrative": full_narrative,
        "causality": "probable",
        "regulatory_category": "requires_hospitalization",
    }
)
obs2, reward, done, info = env.step(action)
check(done, "Episode done")
check(0.0 <= reward.total <= 1.0, f"Reward in [0,1]: {reward.total}")
check(reward.total >= 0.65, f"High score for complete narrative ({reward.total:.3f})")
check(reward.explanation is not None, "Feedback provided")
print(f"  INFO  Score: {reward.total:.3f} | Feedback: {reward.explanation}")
print(f"  INFO  Breakdown: {info.get('breakdown', {})}")

# ===========================================================================
# Task 3: short/incomplete narrative
# ===========================================================================
print("\n" + "="*60)
print("TASK 3: draft_sae_narrative — incomplete narrative (penalty)")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="draft_sae_narrative", scenario_index=0)
env.reset()

action = ClinicalTrialAction(
    action_type="draft_sae_narrative",
    payload={
        "narrative": "Patient had a bad reaction. Drug was stopped.",
        "causality": "unrelated",
        "regulatory_category": "medically_significant",
    }
)
obs2, reward, done, info = env.step(action)
check(reward.total < 0.5, f"Low score for incomplete narrative ({reward.total:.3f})")
print(f"  INFO  Score: {reward.total:.3f}")

# ===========================================================================
# Boundary: step budget exhaustion
# ===========================================================================
print("\n" + "="*60)
print("BOUNDARY: Step budget exhaustion → penalty applied")
print("="*60)

env = ClinicalTrialCoordinatorEnv(task_name="screen_patient", scenario_index=0)
env.reset()

# Burn all steps with info requests
for i in range(4):
    a = ClinicalTrialAction(action_type="request_info", payload={"question": f"Tell me about criterion {i}"})
    obs_t, r, done, info = env.step(a)
    if done:
        break

check(done, "Episode terminates when budget exhausted")
check(r.penalty > 0 or r.total == 0.0, f"Penalty applied or zero reward on exhaustion")
print(f"  INFO  Final reward: {r.total:.3f} | Penalty: {r.penalty}")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "="*60)
if errors:
    print(f"\033[91mFAILED — {len(errors)} checks failed:\033[0m")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print(f"\033[92mALL CHECKS PASSED\033[0m — environment is ready for submission.")
print("="*60 + "\n")