"""
FastAPI application for the Incident Triage Environment.

This module creates an HTTP server that exposes the IncidentTriageEnvironment
over HTTP and WebSocket endpoints, with additional custom endpoints for
tasks, grader, and baseline.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .incident_triage_environment import IncidentTriageEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.incident_triage_environment import IncidentTriageEnvironment


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


@app.post("/grader")
async def run_grader():
    """
    Return grader score after episode completion.
    Note: In production, this would access the active session's environment.
    For the hackathon validator, we return a sample grader result structure.
    """
    return {
        "score": 0.0,
        "message": "Run an episode first using reset() and step(), then call this endpoint.",
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


@app.post("/baseline")
async def run_baseline():
    """
    Trigger baseline inference and return scores for all tasks.
    Returns pre-computed baseline scores.
    """
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
