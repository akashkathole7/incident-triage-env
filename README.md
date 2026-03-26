# 🚨 Incident Triage Environment

**An OpenEnv environment for training and evaluating AI agents on production incident response.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://huggingface.co/openenv)
[![License](https://img.shields.io/badge/license-BSD--3-green)](LICENSE)

## Why This Matters

Every technology company experiences production incidents. On-call SREs must quickly assess severity, investigate root causes, and take remediation actions — often under time pressure at 3 AM. This environment simulates that experience, enabling:

- **Agent evaluation**: Test how well LLM agents can perform operational reasoning
- **Training data**: Generate structured incident response trajectories for RL training
- **Benchmarking**: Compare different agents on realistic, graduated-difficulty incidents

**No equivalent exists in the OpenEnv ecosystem** — this fills a genuine gap in operational AI evaluation.

## Environment Overview

The agent acts as an **on-call SRE** receiving production incidents. Through MCP tools, it can:
1. **Read alerts** — view the initial incident notification
2. **Investigate** — query logs, metrics, traces, and dependency graphs
3. **Classify severity** — assign P1 (critical) through P4 (low) 
4. **Recommend actions** — suggest specific remediation steps
5. **Resolve or escalate** — close the incident or escalate to a specialist team

## Action Space (MCP Tools)

| Tool | Arguments | Description |
|------|-----------|-------------|
| `classify_severity` | `severity: str` (P1/P2/P3/P4) | Classify incident severity |
| `investigate_logs` | `service: str` | View application logs |
| `investigate_metrics` | `service: str` | View performance metrics |
| `investigate_dependencies` | `service: str` | View service dependency graph |
| `investigate_traces` | `service: str` | View distributed traces |
| `recommend_action` | `action_description: str` | Recommend a remediation action |
| `escalate` | `team: str, reason: str` | Escalate to another team (ends episode) |
| `resolve_incident` | `summary: str` | Resolve with summary (ends episode) |

## Observation Space

Each step returns an `Observation` with:
- `done`: Whether the episode has ended
- `reward`: Cumulative reward signal
- `metadata`: Rich context including investigation results, step count, remaining steps

## Tasks (Easy → Medium → Hard)

### Task 1: Alert Classification (Easy)
- **Scenario**: Single clear service alert (e.g., HTTP 500 spike, high latency)
- **Goal**: Classify severity + suggest one remediation action  
- **Max steps**: 8
- **Success criteria**: Correct severity classification and relevant action recommendation

### Task 2: Root Cause Analysis (Medium)
- **Scenario**: Multiple related alerts across 2-3 services
- **Goal**: Find the actual root cause service (not just symptoms) + fix it
- **Max steps**: 12
- **Challenge**: Requires correlating signals and tracing the dependency chain

### Task 3: Cascade Incident (Hard)
- **Scenario**: Complex multi-service outage (5+ services) with **red herring alerts**, contradictory data, and cascading failures
- **Goal**: Full diagnosis + correct escalation decision
- **Max steps**: 18
- **Challenge**: Red herrings (search-service latency is unrelated maintenance), stale database replica promoted during failover, data consistency issues requiring escalation

## Reward Function

| Signal | Reward |
|--------|--------|
| Correct severity classification | +0.3 |
| Partial severity match (±1 level) | +0.1 |
| Useful investigation (new service) | +0.1 |
| Root cause identified | +0.3 |
| Correct remediation action | +0.15 |
| Appropriate escalation decision | +0.2 |
| Per-step time pressure penalty | -0.02 |
| Redundant investigation | -0.05 |
| Invalid/wrong action | -0.1 to -0.15 |

## Grading

Graders are **deterministic** — identical actions always produce identical scores in `[0.0, 1.0]`.

Scoring weights vary by task:
- **Easy**: 45% severity + 30% actions + 25% efficiency
- **Medium**: 25% severity + 35% root cause + 20% actions + 20% efficiency  
- **Hard**: 15% severity + 30% root cause + 20% actions + 20% escalation + 15% efficiency

## Setup & Usage

### Install
```bash
pip install git+https://huggingface.co/spaces/akashkathole/incident_triage_env
```

### Quick Start
```python
from incident_triage_env import IncidentTriageEnv

with IncidentTriageEnv(base_url="https://akashkathole-incident-triage-env.hf.space") as env:
    env.reset(task_id="alert_classification")
    
    # Discover tools
    tools = env.list_tools()
    print([t.name for t in tools])
    
    # Investigate
    result = env.call_tool("investigate_logs", service="payments-service")
    print(result)
    
    # Classify
    result = env.call_tool("classify_severity", severity="P1")
    
    # Resolve
    result = env.call_tool("resolve_incident",
        summary="OOM on payments-service. Recommend restart and memory increase.")
```

### Run Locally with Docker
```bash
# Build
docker build -t incident-triage-env -f server/Dockerfile .

# Run
docker run -p 8000:8000 incident-triage-env

# Test
curl http://localhost:8000/tasks
curl -X POST http://localhost:8000/reset
```

### Run Baseline
```bash
OPENAI_API_KEY=sk-... python baseline.py
```

## Baseline Scores

| Task | Difficulty | gpt-4o-mini Score |
|------|-----------|-------------------|
| Alert Classification | Easy | ~0.65 |
| Root Cause Analysis | Medium | ~0.42 |
| Cascade Incident | Hard | ~0.25 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment for new episode |
| `/step` | POST | Execute an action (MCP tool call) |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List available tasks and action schema |
| `/grader` | POST | Get grader score after episode |
| `/baseline` | POST | Get baseline inference scores |
| `/health` | GET | Health check |

## Project Structure

```
incident_triage_env/
├── __init__.py              # Package exports
├── models.py                # Pydantic models (re-exported from openenv)
├── client.py                # IncidentTriageEnv client
├── baseline.py              # Baseline inference script (OpenAI API)
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Package configuration
├── README.md                # This file
├── data/
│   └── incidents.json       # Incident scenarios (3 tasks, 6 incidents)
├── outputs/
│   ├── logs/
│   └── evals/
└── server/
    ├── incident_triage_environment.py  # MCPEnvironment implementation
    ├── app.py                          # FastAPI app
    ├── requirements.txt               # Server dependencies
    └── Dockerfile                     # Container definition
```

## Built With

- [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Agentic Execution Environments
- [FastAPI](https://fastapi.tiangolo.com/) — Async web framework
- [Pydantic](https://docs.pydantic.dev/) — Type-safe data models
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP tool framework

## License

BSD-3-Clause
