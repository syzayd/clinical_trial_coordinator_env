# ClinicalTrialCoordinatorEnv

**An OpenEnv environment simulating real-world clinical trial coordination workflows.**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-green)](https://huggingface.co/openenv)

---

## Motivation

Clinical trials generate billions in pharmaceutical revenue while directly affecting patient safety. Every trial depends on Clinical Research Coordinators (CRCs) who perform high-stakes, rule-bound tasks: screening hundreds of patients against complex inclusion/exclusion criteria, detecting protocol deviations that could invalidate trial data or harm patients, and drafting regulatory narratives for serious adverse events that must meet FDA/EMA standards.

These tasks are:
- **Real** — performed daily by thousands of healthcare workers globally
- **Structured** — governed by ICH guidelines, FDA regulations, and trial protocols
- **Evaluable** — ground truth is deterministic or semi-deterministic
- **Challenging** — even experienced humans make errors; frontier LLMs currently struggle

This environment gives the RL/agent community a benchmark for healthcare regulatory reasoning that has never existed in standardized form.

---

## Tasks

### Task 1 — Patient Eligibility Screening (Easy)
- **Input:** Patient medical record + trial protocol with inclusion/exclusion criteria
- **Output:** `screen_patient` action with decision, reasoning, and violated criteria
- **Grader:** Deterministic — checks decision correctness, completeness of criterion identification, and reasoning quality
- **Difficulty:** Easy — criteria are explicit in protocol; requires careful cross-referencing
- **Baseline score (Qwen2.5-72B):** ~0.75

### Task 2 — Protocol Deviation Detection (Medium)
- **Input:** Clinical visit record + trial protocol
- **Output:** `flag_deviation` action with deviation type, description, severity, and corrective action
- **Grader:** Deterministic — matches deviation type and severity against ground truth; partial credit for approximate matches
- **Difficulty:** Medium — requires understanding what "required" vs "done" means in clinical context and severity classification
- **Baseline score (Qwen2.5-72B):** ~0.55

### Task 3 — SAE Narrative Drafting (Hard)
- **Input:** Serious Adverse Event report + trial protocol
- **Output:** `draft_sae_narrative` action with ICH E2A compliant narrative, causality assessment, and regulatory category
- **Grader:** Hybrid — deterministic checks for 8 required ICH elements + causality/regulatory classification; narrative quality scoring
- **Difficulty:** Hard — requires regulatory domain knowledge; SAE narratives must cover onset timing, severity, causality chain, concomitant medications, and regulatory criteria
- **Baseline score (Qwen2.5-72B):** ~0.40

---

## Action & Observation Spaces

### Observation Space (`ClinicalTrialObservation`)
```python
task_name: str
step: int
patient: PatientRecord          # demographics, diagnosis, medications, lab values
protocol: TrialProtocol         # inclusion/exclusion criteria, visit schedule, safety thresholds
current_visit: Optional[VisitRecord]   # for deviation detection task
adverse_event: Optional[AdverseEvent]  # for SAE narrative task
audit_log: List[str]            # last 10 actions and results
steps_remaining: int
feedback: str                   # natural language feedback from last step
score_so_far: float
```

### Action Space (`ClinicalTrialAction`)
```python
action_type: str    # "screen_patient" | "flag_deviation" | "draft_sae_narrative" | "request_info" | "finalize"
payload: Dict       # action-specific fields (see below)
```

**screen_patient payload:**
```json
{"decision": "eligible|ineligible", "reason": "...", "criteria_violated": ["..."]}
```

**flag_deviation payload:**
```json
{"deviation_type": "...", "description": "...", "severity": "minor|major|critical", "corrective_action": "..."}
```

**draft_sae_narrative payload:**
```json
{"narrative": "...", "causality": "possible|probable|...", "regulatory_category": "requires_hospitalization|..."}
```

### Reward Space (`ClinicalTrialReward`)
```python
total: float              # combined reward [0.0, 1.0]
correctness: float        # was the core decision right?
completeness: float       # were all required elements present?
reasoning_quality: float  # was the rationale accurate?
efficiency_bonus: float   # bonus for fewer steps used
penalty: float            # deductions for errors
explanation: str          # human-readable breakdown
```

---

## Reward Design

The reward function provides **dense partial credit** throughout the episode:

- `correctness` (50%): Binary yes/no on the main decision (eligible/ineligible, correct deviation type)
- `completeness` (30%): Fraction of required sub-elements correctly identified
- `reasoning_quality` (20%): Presence of key clinical terms and evidence in the rationale
- `efficiency_bonus` (up to +10%): Completing the task correctly in fewer steps
- `penalty` (up to -15%): For step budget exhaustion or invalid action types

This means an agent that gets the decision right but misses half the criteria will score ~0.65, not 1.0. Agents must be thorough.

---

## Setup & Usage

### Local (Docker)

```bash
docker build -t clinical-trial-env .
docker run -p 7860:7860 clinical-trial-env
```

### Direct Python

```bash
pip install -r requirements.txt
python app.py
```

### Run Baseline Inference

```bash
export HF_TOKEN=your_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# All 3 tasks
python inference.py

# Specific task
TASK=detect_deviation python inference.py
```

### Interact with the API

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "screen_patient", "scenario_index": 0}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "YOUR_SESSION_ID",
    "action_type": "screen_patient",
    "payload": {
      "decision": "ineligible",
      "reason": "Patient has NYHA Class III heart failure which is explicitly excluded.",
      "criteria_violated": ["Heart failure (NYHA Class III or IV)"]
    }
  }'
```

---

## Baseline Scores

| Task | Model | Score | Notes |
|------|-------|-------|-------|
| screen_patient | Qwen2.5-72B | ~0.75 | Clear criteria violations, good reasoning |
| detect_deviation | Qwen2.5-72B | ~0.55 | Misses severity nuance on ECG vs PK |
| draft_sae_narrative | Qwen2.5-72B | ~0.40 | Narrative often too brief; causality reasoning weak |

---

## Project Structure

```
clinical-trial-coordinator-env/
├── app.py              FastAPI server (OpenEnv endpoints)
├── env.py              Core environment state machine
├── models.py           Pydantic Observation/Action/Reward models
├── inference.py        Baseline inference script (required)
├── openenv.yaml        OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
├── data/
│   ├── protocols.json        Trial protocol definitions
│   ├── patients.json         Synthetic patient records
│   └── adverse_events.json   Visit records + SAE data
└── graders/
    ├── eligibility_grader.py
    ├── deviation_grader.py
    └── sae_grader.py
```

---

## Extending This Environment

New scenarios can be added to `data/patients.json`, `data/protocols.json`, and `data/adverse_events.json` without any code changes. The environment automatically picks them up.

To add a new task type, extend `TASK_SCENARIOS` in `env.py` and add a grader in `graders/`.

---

## License

MIT
