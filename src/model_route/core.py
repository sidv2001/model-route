"""Shared task, profile, and chosen-action record types."""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

DOMAINS = ("code", "writing", "reasoning")
MAX_RETRIES = 2


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    domain: str
    complexity: int
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("task id must be a nonempty string")
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        if type(self.complexity) is not int or self.complexity not in (1, 2, 3):
            raise ValueError("complexity must be 1, 2, or 3")
        if not isinstance(self.description, str):
            raise ValueError("task description must be a string")


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    credits_per_call: float
    capability_priors: Mapping[str, tuple[float, float, float]]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model name must be a nonempty string")
        if (
            isinstance(self.credits_per_call, bool)
            or not isinstance(self.credits_per_call, (int, float))
            or not isfinite(self.credits_per_call)
            or self.credits_per_call <= 0
        ):
            raise ValueError("credits_per_call must be finite and positive")
        if set(self.capability_priors) != set(DOMAINS):
            raise ValueError(f"capability priors must cover exactly {DOMAINS}")
        for probabilities in self.capability_priors.values():
            if len(probabilities) != 3 or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or not 0 <= value <= 1
                for value in probabilities
            ):
                raise ValueError("each domain needs three probabilities in [0, 1]")

    def prior_success(self, task: Task) -> float:
        return self.capability_priors[task.domain][task.complexity - 1]


@dataclass(frozen=True, slots=True)
class Outcome:
    verified_success: bool | None
    total_credits: float
    retries: int

    def __post_init__(self) -> None:
        if self.verified_success is not None and type(self.verified_success) is not bool:
            raise ValueError("verified_success must be a boolean or None")
        if (
            isinstance(self.total_credits, bool)
            or not isinstance(self.total_credits, (int, float))
            or not isfinite(self.total_credits)
            or self.total_credits <= 0
        ):
            raise ValueError("total_credits must be finite and positive")
        if type(self.retries) is not int or self.retries < 0 or self.retries > MAX_RETRIES:
            raise ValueError(f"retries must be between 0 and {MAX_RETRIES}")


@dataclass(frozen=True, slots=True)
class Decision:
    request_id: int
    task_id: str
    domain: str
    complexity: int
    model: str
    propensity: float


@dataclass(frozen=True, slots=True)
class LoggedDecision:
    task_id: str
    domain: str
    complexity: int
    model: str
    propensity: float
    verified_success: bool | None
    total_credits: float
    retries: int
    reward: float | None


def log_decision(
    task: Task,
    model: str,
    propensity: float,
    outcome: Outcome,
    cost_weight: float,
    retry_weight: float,
) -> LoggedDecision:
    if not isfinite(propensity) or not 0 < propensity <= 1:
        raise ValueError("chosen-action propensity must be in (0, 1]")
    reward = (
        None
        if outcome.verified_success is None
        else (1.0 if outcome.verified_success else 0.0)
        - cost_weight * outcome.total_credits
        - retry_weight * outcome.retries
    )
    return LoggedDecision(
        task_id=task.id,
        domain=task.domain,
        complexity=task.complexity,
        model=model,
        propensity=propensity,
        verified_success=outcome.verified_success,
        total_credits=outcome.total_credits,
        retries=outcome.retries,
        reward=reward,
    )
