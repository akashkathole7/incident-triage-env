"""
Incident Triage Environment Client.

This module provides the client for connecting to an Incident Triage
Environment server. Extends MCPToolClient to provide tool-calling
style interactions for incident investigation.

Example:
    >>> with IncidentTriageEnv(base_url="http://localhost:8000") as env:
    ...     env.reset(task_id="alert_classification")
    ...
    ...     # Discover available tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])
    ...
    ...     # Investigate an incident
    ...     result = env.call_tool("investigate_logs", service="api-gateway")
    ...     print(result)
    ...
    ...     # Classify severity
    ...     result = env.call_tool("classify_severity", severity="P1")
    ...     print(result)
    ...
    ...     # Recommend an action
    ...     result = env.call_tool("recommend_action",
    ...         action_description="restart api-gateway")
    ...     print(result)
    ...
    ...     # Resolve the incident
    ...     result = env.call_tool("resolve_incident",
    ...         summary="Root cause was api-gateway OOM. Restarted service.")
    ...     print(result)
"""

from openenv.core.mcp_client import MCPToolClient


class IncidentTriageEnv(MCPToolClient):
    """
    Client for the Incident Triage Environment.

    Provides a simple interface for interacting with the Incident Triage
    Environment via MCP tools. Inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available investigation tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment (optionally pass task_id)
    - `step(action)`: Execute an action (for advanced use)

    Available tools:
    - classify_severity(severity): Classify incident severity (P1-P4)
    - investigate_logs(service): View logs for a service
    - investigate_metrics(service): View metrics for a service
    - investigate_dependencies(service): View service dependencies
    - investigate_traces(service): View distributed traces
    - recommend_action(action_description): Recommend a fix
    - escalate(team, reason): Escalate to another team
    - resolve_incident(summary): Resolve the incident

    Example:
        >>> with IncidentTriageEnv(base_url="http://localhost:8000") as env:
        ...     env.reset(task_id="root_cause_analysis")
        ...     tools = env.list_tools()
        ...     result = env.call_tool("investigate_logs", service="database-primary")
        ...     result = env.call_tool("classify_severity", severity="P1")
        ...     result = env.call_tool("resolve_incident",
        ...         summary="Database lock contention from analytics query")

    Example with Docker:
        >>> env = IncidentTriageEnv.from_docker_image("incident-triage-env:latest")
        >>> try:
        ...     env.reset()
        ...     tools = env.list_tools()
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
