#!/usr/bin/env python3
"""
Baseline Inference Script for Incident Triage Environment.

This script runs a baseline LLM agent (via OpenAI API) against all 3 tasks
and reports scores. It demonstrates how an LLM agent interacts with the
environment through MCP tools.

Usage:
    OPENAI_API_KEY=sk-... python baseline.py

    # Or with a custom base URL for the environment
    OPENAI_API_KEY=sk-... python baseline.py --env-url http://localhost:8000

Requirements:
    pip install openai httpx
"""

import argparse
import asyncio
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


# System prompt for the baseline SRE agent
SYSTEM_PROMPT = """You are an experienced Site Reliability Engineer (SRE) responding to a production incident.

You have access to the following investigation tools:
- classify_severity(severity): Classify the incident as P1, P2, P3, or P4
- investigate_logs(service): View application logs for a service
- investigate_metrics(service): View performance metrics for a service
- investigate_dependencies(service): View dependency graph for a service
- investigate_traces(service): View distributed traces for a service
- recommend_action(action_description): Recommend a remediation action
- escalate(team, reason): Escalate to another team (ends episode)
- resolve_incident(summary): Resolve the incident (ends episode)

Your approach should be:
1. Read the initial alert carefully
2. Investigate the most likely affected services (logs, metrics, traces)
3. Identify the root cause by following the dependency chain
4. Classify the severity accurately
5. Recommend specific remediation actions
6. Either resolve the incident with a summary or escalate if needed

Be systematic and thorough. Use the investigation tools before jumping to conclusions.
Always end by either resolving or escalating the incident.
"""


def call_tool_via_http(env_url: str, tool_name: str, arguments: dict, session_id: str = "baseline") -> dict:
    """Call an MCP tool on the environment server via HTTP."""
    # Use the step endpoint with CallToolAction
    response = httpx.post(
        f"{env_url}/step",
        json={
            "action": {
                "type": "call_tool",
                "tool_name": tool_name,
                "arguments": arguments,
            }
        },
        params={"session_id": session_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def reset_env(env_url: str, task_id: str, session_id: str = "baseline") -> dict:
    """Reset the environment for a new episode."""
    response = httpx.post(
        f"{env_url}/reset",
        json={"task_id": task_id},
        params={"session_id": session_id},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def run_baseline_for_task(
    client: OpenAI,
    env_url: str,
    task_id: str,
    model: str = "gpt-4o-mini",
    session_id: str = "baseline",
) -> dict:
    """
    Run the baseline agent for a single task.

    Args:
        client: OpenAI client
        env_url: Environment server URL
        task_id: Task to run (alert_classification, root_cause_analysis, cascade_incident)
        model: OpenAI model to use
        session_id: Session ID for the environment

    Returns:
        Result dictionary with score and details
    """
    print(f"\n{'='*60}")
    print(f"Running baseline for task: {task_id}")
    print(f"{'='*60}")

    # Reset environment
    try:
        reset_result = reset_env(env_url, task_id, session_id)
    except Exception as e:
        print(f"  ERROR resetting environment: {e}")
        return {"task_id": task_id, "score": 0.0, "error": str(e)}

    metadata = reset_result.get("metadata", reset_result.get("observation", {}).get("metadata", {}))
    initial_alert = metadata.get("initial_alert", "No alert data")
    affected_services = metadata.get("affected_services", "unknown")
    max_steps = metadata.get("max_steps", 10)

    print(f"  Incident: {metadata.get('incident_title', 'Unknown')}")
    print(f"  Difficulty: {metadata.get('task_difficulty', 'unknown')}")
    print(f"  Max steps: {max_steps}")

    # Build conversation for the LLM
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"NEW INCIDENT ALERT:\n\n{initial_alert}\n\n"
            f"Affected services: {affected_services}\n"
            f"Task: {metadata.get('task_description', 'Investigate and resolve')}\n"
            f"You have {max_steps} steps. Use tools to investigate and resolve.\n\n"
            f"Respond with a JSON object for each action:\n"
            f'{{"tool": "<tool_name>", "args": {{...}}}}\n\n'
            f"Start your investigation."
        )},
    ]

    # Tool definitions for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "classify_severity",
                "description": "Classify incident severity level",
                "parameters": {
                    "type": "object",
                    "properties": {"severity": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]}},
                    "required": ["severity"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "investigate_logs",
                "description": "View application logs for a service",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "investigate_metrics",
                "description": "View performance metrics for a service",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "investigate_dependencies",
                "description": "View dependency graph for a service",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "investigate_traces",
                "description": "View distributed traces for a service",
                "parameters": {
                    "type": "object",
                    "properties": {"service": {"type": "string"}},
                    "required": ["service"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_action",
                "description": "Recommend a remediation action",
                "parameters": {
                    "type": "object",
                    "properties": {"action_description": {"type": "string"}},
                    "required": ["action_description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate",
                "description": "Escalate to another team (ends episode)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "team": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["team", "reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "resolve_incident",
                "description": "Resolve the incident with a summary (ends episode)",
                "parameters": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                },
            },
        },
    ]

    done = False
    step_count = 0

    while not done and step_count < max_steps:
        try:
            # Get LLM response
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,  # Low temperature for deterministic baseline
            )

            message = response.choices[0].message

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    step_count += 1
                    print(f"  Step {step_count}: {tool_name}({tool_args})")

                    # Call the tool on the environment
                    try:
                        result = call_tool_via_http(env_url, tool_name, tool_args, session_id)
                        tool_result = json.dumps(result.get("metadata", result), indent=2)
                        done = result.get("done", False)
                    except Exception as e:
                        tool_result = f"Error calling tool: {e}"

                    # Add tool call and result to messages
                    messages.append({"role": "assistant", "content": None, "tool_calls": [
                        {"id": tool_call.id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}
                    ]})
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

                    if done:
                        break
            else:
                # No tool call — LLM responded with text. Ask it to use a tool.
                messages.append({"role": "assistant", "content": message.content})
                messages.append({"role": "user", "content": "Please use one of the available tools to continue investigation or resolve the incident."})
                step_count += 1

        except Exception as e:
            print(f"  ERROR at step {step_count}: {e}")
            break

    # Get grader result
    try:
        grader_response = httpx.post(
            f"{env_url}/grader",
            params={"session_id": session_id},
            timeout=30.0,
        )
        grader_result = grader_response.json()
    except Exception:
        grader_result = {"score": 0.0}

    score = grader_result.get("score", 0.0)
    print(f"\n  Result: score={score}, steps={step_count}")

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

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Usage: OPENAI_API_KEY=sk-... python baseline.py")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    tasks = ["alert_classification", "root_cause_analysis", "cascade_incident"]
    if args.task:
        tasks = [args.task]

    results = []
    for task_id in tasks:
        result = run_baseline_for_task(
            client=client,
            env_url=args.env_url,
            task_id=task_id,
            model=args.model,
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Task':<25} {'Score':<10} {'Steps':<10}")
    print(f"{'-'*45}")
    for r in results:
        print(f"{r['task_id']:<25} {r['score']:<10.4f} {r['steps']:<10}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0
    print(f"{'-'*45}")
    print(f"{'Average':<25} {avg_score:<10.4f}")
    print(f"\nModel: {args.model}")
    print(f"Environment: {args.env_url}")

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "outputs", "evals")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "baseline_results.json")
    with open(results_path, "w") as f:
        json.dump({"results": results, "model": args.model, "average_score": avg_score}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
