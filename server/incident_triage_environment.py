"""
Incident Triage Environment - MCP Environment Implementation.

A pure MCP environment that simulates production incident triage.
The agent acts as an on-call SRE, classifying severity, investigating
root causes, and recommending remediation actions.

Tools:
- classify_severity(severity): Classify the incident severity (P1-P4)
- investigate_logs(service): View logs for a specific service
- investigate_metrics(service): View metrics for a specific service
- investigate_dependencies(service): View dependency graph for a service
- investigate_traces(service): View distributed traces for a service
- recommend_action(action): Recommend a remediation action
- escalate(team, reason): Escalate to another team with justification
- resolve_incident(summary): Mark incident as resolved with summary
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


# Load incident data
DATA_DIR = Path(__file__).parent.parent / "data"


def _load_incidents() -> dict:
    """Load incident scenarios from JSON file."""
    incidents_path = DATA_DIR / "incidents.json"
    with open(incidents_path, "r") as f:
        return json.load(f)


class IncidentTriageEnvironment(MCPEnvironment):
    """
    Incident Triage Environment - simulates production incident response.

    The agent acts as an on-call SRE and must:
    1. Read alert details and investigate using available tools
    2. Classify incident severity (P1-P4)
    3. Identify root cause through log/metric/trace analysis
    4. Recommend remediation actions
    5. Decide whether to escalate or resolve

    Three tasks with graduated difficulty:
    - alert_classification (Easy): Single-service alert classification
    - root_cause_analysis (Medium): Multi-service root cause investigation
    - cascade_incident (Hard): Complex cascade with red herrings
    """

    def __init__(self):
        """Initialize the incident triage environment with MCP tools."""
        self._incident_data = _load_incidents()
        self._current_task_id: str = "alert_classification"
        self._current_incident: Optional[dict] = None
        self._current_incident_index: int = 0

        # Agent state tracking
        self._agent_severity: Optional[str] = None
        self._agent_root_cause: Optional[str] = None
        self._agent_actions: list[str] = []
        self._agent_escalated: bool = False
        self._agent_escalation_team: Optional[str] = None
        self._agent_escalation_reason: Optional[str] = None
        self._agent_resolved: bool = False
        self._agent_resolve_summary: Optional[str] = None
        self._investigation_history: list[dict] = []
        self._done: bool = False
        self._total_reward: float = 0.0

        # Create MCP server and define tools
        mcp = FastMCP("incident_triage_env")

        env_ref = self  # Reference for closures

        @mcp.tool
        def classify_severity(severity: str) -> str:
            """
            Classify the incident severity level.

            Args:
                severity: The severity level. Must be one of: P1, P2, P3, P4
                    P1 = Critical: Complete service outage or data loss
                    P2 = High: Major feature impaired, workaround possible
                    P3 = Medium: Minor feature impaired, low user impact
                    P4 = Low: Cosmetic issue or improvement request

            Returns:
                Confirmation of classification and reward feedback
            """
            severity = severity.strip().upper()
            valid = ["P1", "P2", "P3", "P4"]
            if severity not in valid:
                env_ref._total_reward -= 0.1
                return f"ERROR: Invalid severity '{severity}'. Must be one of: {valid}"

            env_ref._agent_severity = severity
            env_ref._investigation_history.append({
                "action": "classify_severity",
                "params": {"severity": severity},
                "step": env_ref._state.step_count
            })

            gold = env_ref._current_incident["gold_severity"]
            if severity == gold:
                env_ref._total_reward += 0.3
                return f"Severity classified as {severity}. Classification recorded."
            else:
                severity_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
                diff = abs(severity_order.get(severity, 0) - severity_order.get(gold, 0))
                if diff == 1:
                    env_ref._total_reward += 0.1
                    return f"Severity classified as {severity}. Classification recorded."
                else:
                    env_ref._total_reward -= 0.1
                    return f"Severity classified as {severity}. Classification recorded."

        @mcp.tool
        def investigate_logs(service: str) -> str:
            """
            View application logs for a specific service.

            Args:
                service: Name of the service to investigate (e.g., 'api-gateway', 'payment-service', 'database-primary')

            Returns:
                Recent log entries for the specified service, or an error if service not found
            """
            service = service.strip().lower()
            logs = env_ref._current_incident.get("available_logs", {})

            env_ref._investigation_history.append({
                "action": "investigate_logs",
                "params": {"service": service},
                "step": env_ref._state.step_count
            })

            # Check for duplicate investigation
            prev = [h for h in env_ref._investigation_history[:-1]
                    if h["action"] == "investigate_logs" and h["params"]["service"] == service]
            if prev:
                env_ref._total_reward -= 0.05
                return f"You already investigated logs for {service}. Reviewing again..."

            if service in logs:
                env_ref._total_reward += 0.1
                log_lines = "\n".join(logs[service])
                return f"=== Logs for {service} ===\n{log_lines}"
            else:
                available = list(logs.keys())
                env_ref._total_reward -= 0.05
                return f"Service '{service}' not found. Available services: {available}"

        @mcp.tool
        def investigate_metrics(service: str) -> str:
            """
            View performance metrics for a specific service.

            Args:
                service: Name of the service to investigate (e.g., 'api-gateway', 'payment-service')

            Returns:
                Current performance metrics for the specified service
            """
            service = service.strip().lower()
            metrics = env_ref._current_incident.get("available_metrics", {})

            env_ref._investigation_history.append({
                "action": "investigate_metrics",
                "params": {"service": service},
                "step": env_ref._state.step_count
            })

            prev = [h for h in env_ref._investigation_history[:-1]
                    if h["action"] == "investigate_metrics" and h["params"]["service"] == service]
            if prev:
                env_ref._total_reward -= 0.05
                return f"You already checked metrics for {service}."

            if service in metrics:
                env_ref._total_reward += 0.1
                return f"=== Metrics for {service} ===\n{metrics[service]}"
            else:
                available = list(metrics.keys())
                env_ref._total_reward -= 0.05
                return f"Service '{service}' not found. Available services: {available}"

        @mcp.tool
        def investigate_dependencies(service: str) -> str:
            """
            View the dependency graph for a specific service.

            Args:
                service: Name of the service to check dependencies for

            Returns:
                List of services that this service depends on (downstream dependencies)
            """
            service = service.strip().lower()
            deps = env_ref._current_incident.get("available_dependencies", {})

            env_ref._investigation_history.append({
                "action": "investigate_dependencies",
                "params": {"service": service},
                "step": env_ref._state.step_count
            })

            if service in deps:
                dep_list = deps[service]
                if dep_list:
                    env_ref._total_reward += 0.05
                    return f"=== Dependencies for {service} ===\n{service} depends on: {', '.join(dep_list)}"
                else:
                    env_ref._total_reward += 0.05
                    return f"=== Dependencies for {service} ===\n{service} has no downstream dependencies (leaf service)."
            else:
                available = list(deps.keys())
                return f"Service '{service}' not found. Available services: {available}"

        @mcp.tool
        def investigate_traces(service: str) -> str:
            """
            View distributed traces for a specific service.

            Args:
                service: Name of the service to view traces for

            Returns:
                Recent distributed trace information for the service
            """
            service = service.strip().lower()
            traces = env_ref._current_incident.get("available_traces", {})

            env_ref._investigation_history.append({
                "action": "investigate_traces",
                "params": {"service": service},
                "step": env_ref._state.step_count
            })

            if service in traces:
                env_ref._total_reward += 0.1
                return f"=== Traces for {service} ===\n{traces[service]}"
            else:
                available = list(traces.keys())
                return f"Service '{service}' not found. Available services: {available}"

        @mcp.tool
        def recommend_action(action_description: str) -> str:
            """
            Recommend a remediation action for the incident.

            Args:
                action_description: Description of the recommended action
                    (e.g., 'restart payments-service', 'increase connection pool size',
                     'kill long-running query on database-primary')

            Returns:
                Confirmation that the action recommendation was recorded
            """
            action_description = action_description.strip()
            if not action_description:
                env_ref._total_reward -= 0.1
                return "ERROR: Action description cannot be empty."

            env_ref._agent_actions.append(action_description)
            env_ref._investigation_history.append({
                "action": "recommend_action",
                "params": {"action": action_description},
                "step": env_ref._state.step_count
            })

            # Check if this action matches any gold actions
            gold_actions = env_ref._current_incident.get("gold_actions", [])
            matched = False
            for gold in gold_actions:
                if _fuzzy_match(action_description, gold):
                    matched = True
                    break

            if matched:
                env_ref._total_reward += 0.15
                return f"Action recorded: '{action_description}'. This looks like a relevant remediation step."
            else:
                env_ref._total_reward += 0.02
                return f"Action recorded: '{action_description}'. Noted."

        @mcp.tool
        def escalate(team: str, reason: str) -> str:
            """
            Escalate the incident to another team.

            Args:
                team: The team to escalate to (e.g., 'senior-sre', 'database-team',
                      'security-team', 'payment-team', 'management')
                reason: Justification for why escalation is needed

            Returns:
                Confirmation of escalation. This will end the episode.
            """
            team = team.strip()
            reason = reason.strip()
            if not team or not reason:
                env_ref._total_reward -= 0.1
                return "ERROR: Both team and reason are required for escalation."

            env_ref._agent_escalated = True
            env_ref._agent_escalation_team = team
            env_ref._agent_escalation_reason = reason
            env_ref._done = True

            env_ref._investigation_history.append({
                "action": "escalate",
                "params": {"team": team, "reason": reason},
                "step": env_ref._state.step_count
            })

            gold_escalate = env_ref._current_incident.get("gold_escalate", False)
            if gold_escalate:
                env_ref._total_reward += 0.2
            else:
                env_ref._total_reward -= 0.15

            return f"Incident escalated to {team}. Reason: {reason}. Episode complete."

        @mcp.tool
        def resolve_incident(summary: str) -> str:
            """
            Mark the incident as resolved with a summary of findings and actions taken.

            Args:
                summary: Summary of the incident resolution, including root cause
                         identified and actions taken/recommended

            Returns:
                Confirmation of resolution. This will end the episode.
            """
            summary = summary.strip()
            if not summary:
                env_ref._total_reward -= 0.1
                return "ERROR: Resolution summary cannot be empty."

            env_ref._agent_resolved = True
            env_ref._agent_resolve_summary = summary
            env_ref._done = True

            env_ref._investigation_history.append({
                "action": "resolve_incident",
                "params": {"summary": summary},
                "step": env_ref._state.step_count
            })

            # Check if resolution mentions root cause
            gold_root = env_ref._current_incident.get("gold_root_cause", "")
            gold_detail = env_ref._current_incident.get("gold_root_cause_detail", "")

            if gold_root.lower() in summary.lower() or _fuzzy_match(summary, gold_detail):
                env_ref._agent_root_cause = gold_root
                env_ref._total_reward += 0.3

            gold_escalate = env_ref._current_incident.get("gold_escalate", False)
            if gold_escalate:
                env_ref._total_reward -= 0.15  # Should have escalated

            return f"Incident resolved. Summary recorded. Episode complete."

        # Pass MCP server to base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Optional random seed for incident selection
            episode_id: Optional episode ID
            **kwargs: May include 'task_id' to select specific task

        Returns:
            Initial observation with incident alert details
        """
        # Get task_id from kwargs
        task_id = kwargs.get("task_id", self._current_task_id)
        if task_id not in self._incident_data["tasks"]:
            task_id = "alert_classification"
        self._current_task_id = task_id

        # Select incident
        task_data = self._incident_data["tasks"][task_id]
        incidents = task_data["incidents"]

        if seed is not None:
            idx = seed % len(incidents)
        else:
            idx = self._current_incident_index % len(incidents)
            self._current_incident_index += 1

        self._current_incident = incidents[idx]

        # Reset agent state
        self._agent_severity = None
        self._agent_root_cause = None
        self._agent_actions = []
        self._agent_escalated = False
        self._agent_escalation_team = None
        self._agent_escalation_reason = None
        self._agent_resolved = False
        self._agent_resolve_summary = None
        self._investigation_history = []
        self._done = False
        self._total_reward = 0.0

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Build initial observation
        incident = self._current_incident
        services_list = ", ".join(incident["services"])

        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "status": "active",
                "task_id": task_id,
                "task_name": task_data["name"],
                "task_difficulty": task_data["difficulty"],
                "task_description": task_data["description"],
                "max_steps": task_data["max_steps"],
                "incident_id": incident["incident_id"],
                "incident_title": incident["title"],
                "initial_alert": incident["initial_alert"],
                "affected_services": services_list,
                "instructions": (
                    "You are an on-call SRE responding to this incident. "
                    "Use the available tools to investigate and resolve it. "
                    "Available tools: classify_severity, investigate_logs, "
                    "investigate_metrics, investigate_dependencies, "
                    "investigate_traces, recommend_action, escalate, resolve_incident. "
                    f"You have {task_data['max_steps']} steps to complete this task."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Handle non-MCP actions."""
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use MCP tools: classify_severity, investigate_logs, "
                "investigate_metrics, investigate_dependencies, "
                "investigate_traces, recommend_action, escalate, resolve_incident."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """Execute a step in the environment."""
        self._state.step_count += 1

        # Check max steps
        task_data = self._incident_data["tasks"][self._current_task_id]
        max_steps = task_data["max_steps"]

        if self._state.step_count > max_steps and not self._done:
            self._done = True
            self._total_reward -= 0.2
            return Observation(
                done=True,
                reward=self._total_reward,
                metadata={
                    "status": "timeout",
                    "message": f"Maximum steps ({max_steps}) exceeded. Episode ended.",
                    "total_reward": self._total_reward,
                    "steps_taken": self._state.step_count,
                },
            )

        if self._done:
            return Observation(
                done=True,
                reward=self._total_reward,
                metadata={
                    "status": "completed",
                    "message": "Episode already completed.",
                    "total_reward": self._total_reward,
                },
            )

        # Apply step penalty (time pressure)
        self._total_reward -= 0.02

        # Let base class handle MCP actions
        obs = super().step(action, timeout_s=timeout_s, **kwargs)

        # Return observation with updated state
        return Observation(
            done=self._done,
            reward=self._total_reward,
            metadata={
                **(obs.metadata or {}),
                "status": "completed" if self._done else "active",
                "steps_taken": self._state.step_count,
                "max_steps": max_steps,
                "steps_remaining": max(0, max_steps - self._state.step_count),
                "total_reward": round(self._total_reward, 4),
                "severity_classified": self._agent_severity is not None,
                "actions_recommended": len(self._agent_actions),
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def get_grader_result(self) -> dict:
        """
        Compute the deterministic grader score for the current episode.

        Returns:
            Dictionary with score (0.0-1.0) and breakdown
        """
        if self._current_incident is None:
            return {"score": 0.0, "breakdown": {}, "error": "No episode run"}

        incident = self._current_incident
        task_id = self._current_task_id

        # Severity accuracy
        severity_score = 0.0
        if self._agent_severity:
            severity_order = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}
            gold_val = severity_order.get(incident["gold_severity"], 0)
            agent_val = severity_order.get(self._agent_severity, 0)
            diff = abs(gold_val - agent_val)
            if diff == 0:
                severity_score = 1.0
            elif diff == 1:
                severity_score = 0.5
            else:
                severity_score = 0.0

        # Root cause accuracy
        root_cause_score = 0.0
        gold_root = incident["gold_root_cause"].lower()
        gold_detail = incident.get("gold_root_cause_detail", "").lower()

        # Check from resolve summary or investigation history
        summary = (self._agent_resolve_summary or "").lower()
        if gold_root in summary:
            root_cause_score = 1.0
        elif any(word in summary for word in gold_detail.split() if len(word) > 5):
            root_cause_score = 0.5

        # Action quality
        action_score = 0.0
        gold_actions = incident.get("gold_actions", [])
        if gold_actions and self._agent_actions:
            matches = 0
            for agent_act in self._agent_actions:
                for gold_act in gold_actions:
                    if _fuzzy_match(agent_act, gold_act):
                        matches += 1
                        break
            action_score = min(1.0, matches / len(gold_actions))

        # Escalation decision (for hard tasks)
        escalation_score = 0.0
        gold_escalate = incident.get("gold_escalate", False)
        if gold_escalate and self._agent_escalated:
            escalation_score = 1.0
        elif not gold_escalate and not self._agent_escalated:
            escalation_score = 1.0
        else:
            escalation_score = 0.0

        # Efficiency (how many steps used vs max)
        task_data = self._incident_data["tasks"][task_id]
        max_steps = task_data["max_steps"]
        steps = self._state.step_count
        efficiency_score = max(0.0, 1.0 - (steps / max_steps) * 0.5) if max_steps > 0 else 0.0

        # Weighted score per task
        if task_id == "alert_classification":
            score = (
                0.45 * severity_score +
                0.30 * action_score +
                0.25 * efficiency_score
            )
        elif task_id == "root_cause_analysis":
            score = (
                0.25 * severity_score +
                0.35 * root_cause_score +
                0.20 * action_score +
                0.20 * efficiency_score
            )
        else:  # cascade_incident
            score = (
                0.15 * severity_score +
                0.30 * root_cause_score +
                0.20 * action_score +
                0.20 * escalation_score +
                0.15 * efficiency_score
            )

        score = round(max(0.0, min(1.0, score)), 4)

        return {
            "score": score,
            "task_id": task_id,
            "incident_id": incident["incident_id"],
            "breakdown": {
                "severity_accuracy": round(severity_score, 4),
                "root_cause_accuracy": round(root_cause_score, 4),
                "action_quality": round(action_score, 4),
                "escalation_decision": round(escalation_score, 4),
                "efficiency": round(efficiency_score, 4),
            },
            "agent_summary": {
                "severity": self._agent_severity,
                "root_cause": self._agent_root_cause,
                "actions": self._agent_actions,
                "escalated": self._agent_escalated,
                "resolved": self._agent_resolved,
                "steps_taken": self._state.step_count,
            },
        }

    def get_tasks(self) -> list[dict]:
        """Return list of available tasks with descriptions."""
        tasks = []
        for task_id, task_data in self._incident_data["tasks"].items():
            tasks.append({
                "task_id": task_id,
                "name": task_data["name"],
                "difficulty": task_data["difficulty"],
                "description": task_data["description"],
                "max_steps": task_data["max_steps"],
                "num_incidents": len(task_data["incidents"]),
                "action_schema": {
                    "tools": [
                        "classify_severity(severity: str)",
                        "investigate_logs(service: str)",
                        "investigate_metrics(service: str)",
                        "investigate_dependencies(service: str)",
                        "investigate_traces(service: str)",
                        "recommend_action(action_description: str)",
                        "escalate(team: str, reason: str)",
                        "resolve_incident(summary: str)",
                    ]
                },
            })
        return tasks


def _fuzzy_match(candidate: str, reference: str) -> bool:
    """
    Simple fuzzy matching - checks if key words from reference appear in candidate.
    Deterministic: no randomness, no ML, pure string matching.
    """
    candidate_lower = candidate.lower()
    reference_lower = reference.lower()

    # Direct substring match
    if reference_lower in candidate_lower or candidate_lower in reference_lower:
        return True

    # Key word overlap
    ref_words = set(w for w in reference_lower.split() if len(w) > 3)
    cand_words = set(w for w in candidate_lower.split() if len(w) > 3)

    if not ref_words:
        return False

    overlap = len(ref_words & cand_words)
    return overlap >= max(1, len(ref_words) * 0.4)
