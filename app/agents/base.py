"""
Base agent.

Every agent is a small, single-responsibility unit. Keeping them thin makes
the pipeline easy to test and easy to explain in an interview: each agent does
one thing and hands a typed object to the next.

If you build on CrewAI (as the resume states), you can wrap each of these
`run` methods as a CrewAI Agent/Task. The scaffold keeps them as plain classes
so the control flow is explicit and debuggable first.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Do the agent's single job and return a typed result."""
        raise NotImplementedError
