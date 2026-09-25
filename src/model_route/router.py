"""Categorical contextual epsilon-greedy routing with opt-in user feedback."""

from math import isfinite
from random import Random
from typing import Sequence

from .core import (
    MAX_RETRIES,
    Decision,
    LoggedDecision,
    ModelProfile,
    Outcome,
    Task,
    log_decision,
)


def _finite_in_range(value: float, low: float, high: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be finite and between {low} and {high}")


def _user_id(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("user_id must be a nonempty string")


class ContextualRouter:
    def __init__(
        self,
        profiles: Sequence[ModelProfile],
        *,
        epsilon: float = 0.15,
        prior_strength: float = 4.0,
        cost_weight: float = 0.06,
        retry_weight: float = 0.03,
        max_retries: int = 1,
        seed: int = 7,
    ) -> None:
        self.profiles = tuple(profiles)
        if not self.profiles or len({profile.name for profile in self.profiles}) != len(self.profiles):
            raise ValueError("router requires distinct model profiles")
        _finite_in_range(epsilon, 0, 1, "epsilon")
        if (
            isinstance(prior_strength, bool)
            or not isinstance(prior_strength, (int, float))
            or not isfinite(prior_strength)
            or prior_strength <= 0
        ):
            raise ValueError("prior_strength must be finite and positive")
        for name, value in (("cost_weight", cost_weight), ("retry_weight", retry_weight)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES}")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.epsilon = epsilon
        self.prior_strength = prior_strength
        self.cost_weight = cost_weight
        self.retry_weight = retry_weight
        self.max_retries = max_retries
        self._rng = Random(seed)
        self._by_name = {profile.name: profile for profile in self.profiles}
        self._consenting_users: set[str] = set()
        self._history: dict[str, list[LoggedDecision]] = {}
        self._pending: dict[int, tuple[Decision, str | None]] = {}
        self._next_request_id = 1

    def opt_in(self, user_id: str) -> None:
        _user_id(user_id)
        self._consenting_users.add(user_id)
        self._history.setdefault(user_id, [])

    def forget(self, user_id: str) -> None:
        _user_id(user_id)
        self._consenting_users.discard(user_id)
        self._history.pop(user_id, None)
        for request_id, (decision, pending_user) in self._pending.items():
            if pending_user == user_id:
                self._pending[request_id] = (decision, None)

    def history_for(self, user_id: str) -> tuple[LoggedDecision, ...]:
        _user_id(user_id)
        return tuple(self._history.get(user_id, ()))

    def estimate(self, task: Task, model: str, user_id: str | None = None) -> float:
        if model not in self._by_name:
            raise ValueError(f"unknown model: {model}")
        if user_id is not None:
            _user_id(user_id)
        profile = self._by_name[model]
        success_per_attempt = profile.prior_success(task)
        expected_calls = sum(
            (1 - success_per_attempt) ** attempt
            for attempt in range(self.max_retries + 1)
        )
        prior_reward = (
            1 - (1 - success_per_attempt) ** (self.max_retries + 1)
            - self.cost_weight * profile.credits_per_call * expected_calls
            - self.retry_weight * (expected_calls - 1)
        )
        if user_id not in self._consenting_users:
            return prior_reward
        observations = [
            entry.reward
            for entry in self._history[user_id]
            if entry.domain == task.domain
            and entry.complexity == task.complexity
            and entry.model == model
            and entry.reward is not None
        ]
        return (self.prior_strength * prior_reward + sum(observations)) / (
            self.prior_strength + len(observations)
        )

    def action_probabilities(self, task: Task, user_id: str | None = None) -> dict[str, float]:
        greedy = max(
            self.profiles,
            key=lambda profile: self.estimate(task, profile.name, user_id),
        ).name
        explore = self.epsilon / len(self.profiles)
        return {
            profile.name: explore + (1 - self.epsilon if profile.name == greedy else 0)
            for profile in self.profiles
        }

    def select(self, task: Task, user_id: str | None = None) -> Decision:
        probabilities = self.action_probabilities(task, user_id)
        greedy = max(self.profiles, key=lambda profile: probabilities[profile.name]).name
        model = (
            self._rng.choice(self.profiles).name
            if self._rng.random() < self.epsilon
            else greedy
        )
        decision = Decision(
            request_id=self._next_request_id,
            task_id=task.id,
            domain=task.domain,
            complexity=task.complexity,
            model=model,
            propensity=probabilities[model],
        )
        self._pending[self._next_request_id] = (
            decision,
            user_id if user_id in self._consenting_users else None,
        )
        self._next_request_id += 1
        return decision

    def observe(self, decision: Decision, outcome: Outcome) -> LoggedDecision:
        pending = self._pending.get(decision.request_id)
        if pending is None or pending[0] != decision:
            raise ValueError("unknown or already observed decision")
        if outcome.retries > self.max_retries:
            raise ValueError("outcome exceeds configured retry limit")
        entry = log_decision(
            Task(decision.task_id, decision.domain, decision.complexity),
            decision.model,
            decision.propensity,
            outcome,
            self.cost_weight,
            self.retry_weight,
        )
        del self._pending[decision.request_id]
        pending_user = pending[1]
        if pending_user is not None and outcome.verified_success is not None:
            self._history[pending_user].append(entry)
        return entry
