"""
Baseline inference script for ClinicalTrialCoordinatorEnv.

Follows the EXACT [START] / [STEP] / [END] log format required by the hackathon.
Uses OpenAI client with API_BASE_URL / MODEL_NAME / HF_TOKEN environment variables.

Usage:
    python inference.py
    TASK=detect_deviation python inference.py
    TASK=draft_sae_narrative python inference.py
"""
import json
import os
import sys
import textwrap
import uuid
from typing import Dict, List, Optional, Any

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "hf-no-key"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("TASK", "screen_patient")
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "http://localhost:7860"
BENCHMARK = "clinical_trial_coordinator_env"

MAX_STEPS = 8
TEMPERATURE = 0.3
MAX_TOKENS = 800
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging (REQUIRED FORMAT — do not change field names or order)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Env HTTP client
# ---------------------------------------------------------------------------

def env_reset(task_name: str, scenario_index: int = 0) -> Dict:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={
        "task_name": task_name,
        "scenario_index": scenario_index,
    })
    resp.raise_for_status()
    return resp.json()


def env_step(session_id: str, action_type: str, payload: Dict) -> Dict:
    resp = requests.post(f"{ENV_BASE_URL}/step", json={
        "session_id": session_id,
        "action_type": action_type,
        "payload": payload,
    })
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "screen_patient": textwrap.dedent("""
        You are an expert clinical research coordinator reviewing patient eligibility for a clinical trial.

        You will receive:
        - A patient record (demographics, diagnosis, medications, lab values)
        - A trial protocol (inclusion/exclusion criteria)

        Your task: Determine if the patient is eligible or ineligible.

        Respond with ONLY a JSON object in this exact format:
        {
          "decision": "eligible" or "ineligible",
          "reason": "detailed explanation referencing specific lab values and criteria",
          "criteria_violated": ["list of exclusion criteria violated, if any"]
        }

        Be precise. Reference exact lab values (e.g., "eGFR 38 which is above the minimum of 30").
        If ineligible, list ALL violated exclusion criteria.
        No preamble, no markdown — just the JSON.
    """).strip(),

    "detect_deviation": textwrap.dedent("""
        You are an expert clinical research coordinator reviewing a clinical visit for protocol deviations.

        You will receive:
        - A visit record (what was done, what was scheduled, timing)
        - The trial protocol

        Your task: Identify the most significant protocol deviation.

        Respond with ONLY a JSON object in this exact format:
        {
          "deviation_type": "missing_procedure" or "visit_timing" or "prohibited_medication" or "safety_threshold",
          "description": "specific description of what deviated and by how much",
          "severity": "minor" or "major" or "critical",
          "corrective_action": "what should be done to address this deviation"
        }

        Be specific — name the exact procedure, visit, or value that deviated.
        No preamble, no markdown — just the JSON.
    """).strip(),

    "draft_sae_narrative": textwrap.dedent("""
        You are a clinical research coordinator drafting a Serious Adverse Event (SAE) narrative
        following ICH E2A guidelines for regulatory submission.

        You will receive details of a serious adverse event.

        Your task: Write a complete SAE narrative.

        Respond with ONLY a JSON object in this exact format:
        {
          "narrative": "Full ICH E2A narrative (minimum 150 words) covering: patient identifier, event description, onset date/time, severity, prior and concomitant medications, relevant medical history, timeline, action taken, causality assessment rationale, and outcome",
          "causality": "unrelated" or "possible" or "probable" or "definite",
          "regulatory_category": "requires_hospitalization" or "life-threatening" or "results_in_death" or "persistent_disability" or "congenital_anomaly" or "medically_significant"
        }

        The narrative must be detailed enough for regulatory submission. Include the patient ID,
        specific event, onset day relative to study drug start, grade/severity, all actions taken,
        and your causality reasoning.
        No preamble, no markdown — just the JSON.
    """).strip(),
}

# ---------------------------------------------------------------------------
# Agent: get model action
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    task_name: str,
    observation: Dict,
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Call the LLM and parse its JSON response into an action payload."""
    system_prompt = SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["screen_patient"])

    # Build user prompt from observation
    obs_summary = json.dumps(observation, indent=2, default=str)
    history_block = "\n".join(history[-3:]) if history else "None"

    user_prompt = textwrap.dedent(f"""
        Step: {step}
        Previous feedback: {observation.get('feedback', 'None')}
        Steps remaining: {observation.get('steps_remaining', '?')}

        Current observation:
        {obs_summary}

        History of actions:
        {history_block}

        Provide your action now.
    """).strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback actions
        fallbacks = {
            "screen_patient": {"decision": "ineligible", "reason": "Unable to determine — defaulting to caution.", "criteria_violated": []},
            "detect_deviation": {"deviation_type": "missing_procedure", "description": "Unable to determine.", "severity": "minor", "corrective_action": "Review visit record."},
            "draft_sae_narrative": {"narrative": "SAE occurred. Patient experienced a serious adverse event.", "causality": "possible", "regulatory_category": "requires_hospitalization"},
        }
        return fallbacks.get(task_name, {})


ACTION_TYPE_MAP = {
    "screen_patient": "screen_patient",
    "detect_deviation": "flag_deviation",
    "draft_sae_narrative": "draft_sae_narrative",
}

# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task_name: str, scenario_index: int = 0) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Reset
    reset_data = env_reset(task_name, scenario_index)
    session_id = reset_data["session_id"]
    observation = reset_data["observation"]

    history: List[str] = []
    rewards: List[float] = []
    final_score = 0.0

    for step_num in range(1, MAX_STEPS + 1):
        # Get model action
        payload = get_model_action(client, task_name, observation, step_num, history)
        action_type = ACTION_TYPE_MAP.get(task_name, "screen_patient")
        history.append(f"Step {step_num}: {json.dumps(payload)[:100]}")

        # Execute step
        error = None
        try:
            step_data = env_step(session_id, action_type, payload)
        except Exception as exc:
            error = str(exc)
            log_step(step_num, action_type, 0.0, False, error)
            break

        reward_val = step_data["reward"]["total"]
        done = step_data["done"]
        rewards.append(reward_val)
        final_score = reward_val

        log_step(step_num, action_type, reward_val, done, error)

        if done:
            observation = step_data["observation"]
            break
        observation = step_data["observation"]

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=len(rewards), score=final_score, rewards=rewards)
    return final_score


# ---------------------------------------------------------------------------
# Main: run all 3 tasks
# ---------------------------------------------------------------------------

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        ("screen_patient", 0),     # Task 1: Easy
        ("detect_deviation", 0),   # Task 2: Medium
        ("draft_sae_narrative", 0), # Task 3: Hard
    ]

    # If TASK env var is set, only run that one
    env_task = os.getenv("TASK")
    if env_task:
        tasks = [(env_task, 0)]

    all_scores = []
    for task_name, scenario_idx in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_name} (scenario {scenario_idx})", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_episode(client, task_name, scenario_idx)
        all_scores.append(score)

    if len(all_scores) > 1:
        avg = sum(all_scores) / len(all_scores)
        print(f"\n[SUMMARY] Tasks completed: {len(all_scores)}", flush=True)
        print(f"[SUMMARY] Scores: {[f'{s:.3f}' for s in all_scores]}", flush=True)
        print(f"[SUMMARY] Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
