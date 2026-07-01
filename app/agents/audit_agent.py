"""
Audit Agent.

Job: produce a traceable log of everything the pipeline did for a request, so
any answer can be reconstructed and defended later ("why did the system say
this?"). In a regulated / policy setting this trail is the whole point.

Scaffold behaviour: collects AuditEvents in memory and can append them to a
JSONL file. TODO: ship them to your real sink (Langfuse, a DB, or object store).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents.base import BaseAgent
from app.models.schemas import AuditEvent


class AuditAgent(BaseAgent):
    name = "audit_agent"

    def __init__(self, sink_path: str | None = "data/audit_log.jsonl"):
        self.sink_path = sink_path
        self.events: list[AuditEvent] = []

    def record(self, agent: str, action: str, **payload: Any) -> AuditEvent:
        event = AuditEvent(agent=agent, action=action, payload=payload)
        self.events.append(event)
        return event

    def flush(self) -> list[AuditEvent]:
        """Persist collected events and return them. Called once per request."""
        if self.sink_path:
            path = Path(self.sink_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                for e in self.events:
                    fh.write(json.dumps(e.model_dump(), default=str) + "\n")
        collected = list(self.events)
        self.events.clear()
        return collected

    # BaseAgent contract; the audit agent is event-driven rather than one-shot.
    def run(self, *args: Any, **kwargs: Any) -> list[AuditEvent]:
        return self.flush()
