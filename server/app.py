"""
FastAPI application for the Incident Triage Environment.

This module creates an HTTP server that exposes the IncidentTriageEnvironment
over HTTP and WebSocket endpoints, with additional custom endpoints for
tasks, grader, and baseline.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .incident_triage_environment import IncidentTriageEnvironment
except ImportError:
    from server.incident_triage_environment import IncidentTriageEnvironment


# Shared environment registry for session-to-env mapping
# This lets custom endpoints access the same env instances as the MCP server
_env_registry: dict[str, IncidentTriageEnvironment] = {}


def _env_factory():
    """Factory that creates environments and registers them."""
    env = IncidentTriageEnvironment()
    return env


# Create the app with MCP types
app = create_app(
    IncidentTriageEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="incident_triage_env",
)


# ---- Custom Endpoints ----

@app.get("/tasks")
async def get_tasks():
    """Return list of available tasks and action schema."""
    env = IncidentTriageEnvironment()
    return {"tasks": env.get_tasks()}


@app.post("/reset_env")
async def reset_env(body: dict = {}):
    """
    Custom reset endpoint that returns full incident context.

    Unlike the standard /reset (which returns Observation through MCP
    serialization), this endpoint returns all incident metadata directly
    in the JSON response so HTTP-only clients (like baseline scripts)
    can see the alert details.

    Args (JSON body):
        task_id (str): One of 'alert_classification', 'root_cause_analysis', 'cascade_incident'
        seed (int): Optional seed for incident selection

    Returns:
        Full reset result including initial_alert, affected_services, etc.
    """
    task_id = body.get("task_id", "alert_classification")
    seed = body.get("seed", None)

    env = IncidentTriageEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    _env_registry[task_id] = env  # Store for grader access

    return {
        "done": obs.done,
        "reward": obs.reward,
        "metadata": obs.metadata,
    }


@app.post("/step_env")
async def step_env(body: dict = {}):
    """
    Custom step endpoint that returns full observation metadata.

    Args (JSON body):
        task_id (str): Task being run (to look up stored env)
        tool_name (str): MCP tool to call
        arguments (dict): Arguments for the tool

    Returns:
        Full step result including tool output text, reward, done, metadata
    """
    task_id = body.get("task_id", "alert_classification")
    tool_name = body.get("tool_name", "")
    arguments = body.get("arguments", {})

    env = _env_registry.get(task_id)
    if env is None:
        return {"error": f"No active session for task '{task_id}'. Call /reset_env first."}

    # Call the MCP tool directly using await (we're inside an async handler)
    from fastmcp import Client

    try:
        async with Client(env._mcp) as client:
            result = await client.call_tool(tool_name, arguments)
        tool_output = "\n".join(str(c.text) if hasattr(c, 'text') else str(c) for c in result)
    except Exception as e:
        tool_output = f"Error: {e}"

    # Manually increment step counter for tracking
    env._state.step_count += 1

    return {
        "done": env._done,
        "reward": round(env._total_reward, 4),
        "tool_output": tool_output,

        "metadata": {
            "steps_taken": env._state.step_count,
            "severity_classified": env._agent_severity is not None,
            "actions_recommended": len(env._agent_actions),
            "escalated": env._agent_escalated,
            "resolved": env._agent_resolved,
        }
    }


@app.post("/grader")
async def run_grader(body: dict = {}):
    """
    Return grader score after episode completion.

    Args (JSON body):
        task_id (str): Task to get grader result for

    Returns:
        Deterministic grader result with score in [0.0, 1.0] and breakdown
    """
    task_id = body.get("task_id", "alert_classification")
    env = _env_registry.get(task_id)

    if env is None:
        return {
            "score": 0.0,
            "message": "No active session. Call /reset_env first, run steps, then call /grader.",
            "schema": {
                "score": "float in [0.0, 1.0]",
                "breakdown": {
                    "severity_accuracy": "float",
                    "root_cause_accuracy": "float",
                    "action_quality": "float",
                    "escalation_decision": "float",
                    "efficiency": "float",
                },
            },
        }

    return env.get_grader_result()


@app.post("/baseline")
async def run_baseline():
    """Return pre-computed baseline scores for all tasks."""
    return {
        "baseline_scores": {
            "alert_classification": {
                "score": 0.65,
                "difficulty": "easy",
                "description": "Classify severity and suggest fix for single-service alerts",
            },
            "root_cause_analysis": {
                "score": 0.42,
                "difficulty": "medium",
                "description": "Find root cause across multiple correlated services",
            },
            "cascade_incident": {
                "score": 0.25,
                "difficulty": "hard",
                "description": "Diagnose complex cascade with red herrings and escalation",
            },
        },
        "agent": "gpt-4o-mini baseline",
        "note": "Run baseline.py with OPENAI_API_KEY for live scores",
    }


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
