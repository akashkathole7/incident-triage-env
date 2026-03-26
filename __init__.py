"""
Incident Triage Environment - Production incident response simulation.

This environment exposes all functionality through MCP tools:
- `classify_severity(severity)`: Classify incident severity (P1-P4)
- `investigate_logs(service)`: View application logs for a service
- `investigate_metrics(service)`: View performance metrics
- `investigate_dependencies(service)`: View service dependency graph
- `investigate_traces(service)`: View distributed traces
- `recommend_action(action_description)`: Recommend remediation
- `escalate(team, reason)`: Escalate to another team
- `resolve_incident(summary)`: Resolve the incident

Example:
    >>> from incident_triage_env import IncidentTriageEnv
    >>>
    >>> with IncidentTriageEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task_id="alert_classification")
    ...     tools = env.list_tools()
    ...     result = env.call_tool("classify_severity", severity="P1")
    ...     result = env.call_tool("resolve_incident",
    ...         summary="OOM on payments-service, restarted")
"""

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import IncidentTriageEnv

__all__ = ["IncidentTriageEnv", "CallToolAction", "ListToolsAction"]
