#!/usr/bin/env python3
"""
Baseline Inference Script for Incident Triage Environment.

Runs a baseline LLM agent (via OpenAI API) against all 3 tasks
and reports grader scores. Uses the custom /reset_env and /step_env
endpoints which return full observation metadata.

Usage:
    set OPENAI_API_KEY=sk-...
    python baseline.py

    # Or with custom environment URL
    python baseline.py --env-url http://localhost:8000
"""

import argparse
import json
import os
import sys
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package required. Install with: pip install openai")
    sys.exit(1)

try:
    import httpx
except ImportError:
    print("ERROR: httpx package required. Install with: pip install httpx")
    sys.exit(1)


SYSTEM_PROMPT = """You are an experienced Site Reliability Engineer (SRE) responding to a production incident.

You have access to these investigation tools:
- classify_severity(severity): Classify the incident as P1, P2, P3, or P4
- investigate_logs(service): View application logs for a service
- investigate_metrics(service): View performance metrics for a service
- investigate_dependencies(service): View dependency graph for a service
- investigate_traces(service): View distributed traces for a service
- recommend_action(action_description): Recommend a remediation action
- escalate(team, reason): Escalate to another team (ends episode)
- resolve_incident(summary): Resolve with findings summary (ends episode)

Approach:
1. Read the alert carefully and note which services are affected
2. Investigate the most critical services first (logs → metrics → traces)
3. Follow the dependency chain to find the ROOT CAUSE (not just symptoms)
4. Classify severity accurately
5. Recommend specific remediation actions
6. Resolve the incident with a summary mentioning the root cause service, OR escalate if the situation requires it

IMPORTANT: Be systematic. Don't guess — investigate first. Always end by resolving or escalating."""


# OpenAI tool definitions that map to our environment's MCP tools
TOOLS = [
    {"type": "function", "function": {
        "name": "classify_severity",
        "description": "Classify incident severity level",
        "parameters": {"type": "object", "properties": {"severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]}}, "required": ["severity"]},
    }},
    {"type": "function", "function": {
        "name": "investigate_logs",
        "description": "View application logs for a service",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "investigate_metrics",
        "description": "View performance metrics for a service",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "investigate_dependencies",
        "description": "View dependency graph for a service",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "investigate_traces",
        "description": "View distributed traces for a service",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
    }},
    {"type": "function", "function": {
        "name": "recommend_action",
        "description": "Recommend a remediation action",
        "parameters": {"type": "object", "properties": {"action_description": {"type": "string"}}, "required": ["action_description"]},
    }},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate to another team (ends the episode)",
        "parameters": {"type": "object", "properties": {"team": {"type": "string"}, "reason": {"type": "string"}}, "required": ["team", "reason"]},
    }},
    {"type": "function", "function": {
        "name": "resolve_incident",
        "description": "Resolve the incident with a summary of findings (ends the episode)",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    }},
]


def run_task(client: OpenAI, env_url: str, task_id: str, model: str = "gpt-4o-mini") -> dict:
    """Run the baseline agent for a single task."""

    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    # 1. Reset environment via custom endpoint
    r = httpx.post(f"{env_url}/reset_env", json={"task_id": task_id}, timeout=30)
    if r.status_code != 200:
        print(f"  ERROR: Reset failed ({r.status_code})")
        return {"task_id": task_id, "score": 0.0, "error": f"Reset failed: {r.status_code}"}

    reset_data = r.json()
    metadata = reset_data.get("metadata", {})
    initial_alert = metadata.get("initial_alert", "No alert received")
    affected = metadata.get("affected_services", "unknown")
    max_steps = metadata.get("max_steps", 10)
    task_desc = metadata.get("task_description", "Investigate and resolve")

    print(f"  Incident: {metadata.get('incident_title', 'Unknown')}")
    print(f"  Difficulty: {metadata.get('task_difficulty', '?')}")
    print(f"  Max steps: {max_steps}")

    # 2. Build LLM conversation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"INCOMING INCIDENT ALERT\n"
            f"{'='*40}\n\n"
            f"{initial_alert}\n\n"
            f"Affected services: {affected}\n"
            f"Task objective: {task_desc}\n"
            f"Steps available: {max_steps}\n\n"
            f"Begin your investigation now."
        )},
    ]

    # 3. Agent loop
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    step_count += 1
                    print(f"  Step {step_count}: {tool_name}({json.dumps(tool_args)[:80]})")

                    # Call through custom endpoint
                    step_r = httpx.post(f"{env_url}/step_env", json={
                        "task_id": task_id,
                        "tool_name": tool_name,
                        "arguments": tool_args,
                    }, timeout=30)

                    if step_r.status_code == 200:
                        step_data = step_r.json()
                        tool_output = step_data.get("tool_output", "No output")
                        done = step_data.get("done", False)
                        reward = step_data.get("reward", 0.0)
                        print(f"         → reward={reward}, done={done}")
                    else:
                        tool_output = f"Error: HTTP {step_r.status_code}"

                    # Append to conversation
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)}
                        }]
                    })
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_output})

                    if done:
                        break
            else:
                # LLM gave text instead of tool call — nudge it
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": "Please use a tool to continue your investigation or resolve/escalate the incident."})
                step_count += 1

        except Exception as e:
            print(f"  ERROR at step {step_count}: {e}")
            break

    # 4. Get grader score
    grader_r = httpx.post(f"{env_url}/grader", json={"task_id": task_id}, timeout=30)
    grader_result = grader_r.json() if grader_r.status_code == 200 else {"score": 0.0}

    score = grader_result.get("score", 0.0)
    breakdown = grader_result.get("breakdown", {})

    print(f"\n  SCORE: {score}")
    if breakdown:
        for k, v in breakdown.items():
            print(f"    {k}: {v}")

    return {
        "task_id": task_id,
        "score": score,
        "steps": step_count,
        "grader_result": grader_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline inference for Incident Triage Environment")
    parser.add_argument("--env-url", default="http://localhost:8000", help="Environment server URL")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--task", default=None, help="Run specific task only")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    tasks = ["alert_classification", "root_cause_analysis", "cascade_incident"]
    if args.task:
        tasks = [args.task]

    results = []
    for task_id in tasks:
        result = run_task(client, args.env_url, task_id, args.model)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Score':<10} {'Steps':<10}")
    print(f"{'-'*45}")
    for r in results:
        print(f"{r['task_id']:<25} {r['score']:<10.4f} {r['steps']:<10}")
    avg = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"{'-'*45}")
    print(f"{'Average':<25} {avg:<10.4f}")
    print(f"\nModel: {args.model}")

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), "outputs", "evals")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_results.json")
    with open(out_path, "w") as f:
        json.dump({"results": results, "model": args.model, "average_score": avg}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
