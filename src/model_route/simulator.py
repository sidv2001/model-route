"""Synthetic oracle; only the selected model is called and exposed."""

from hashlib import sha256
from typing import Sequence

from .core import MAX_RETRIES, ModelProfile, Outcome, Task
from .fixtures import Scenario


class SyntheticEnvironment:
    def __init__(
        self, profiles: Sequence[ModelProfile], scenario: Scenario, *, seed: int = 7
    ) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._profiles = {profile.name: profile for profile in profiles}
        if not self._profiles or len(self._profiles) != len(profiles):
            raise ValueError("environment requires distinct model profiles")
        self._scenario = scenario
        self._seed = seed

    def _draw(self, task_id: str, attempt: int) -> float:
        digest = sha256(f"{self._seed}:{task_id}:{attempt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def run(
        self,
        task: Task,
        user_id: str,
        model: str,
        *,
        max_retries: int,
        verified: bool = True,
    ) -> Outcome:
        if model not in self._profiles:
            raise ValueError(f"unknown model: {model}")
        if user_id not in self._scenario.success_offsets:
            raise ValueError(f"unknown synthetic user: {user_id}")
        if type(max_retries) is not int or not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES}")
        if type(verified) is not bool:
            raise ValueError("verified must be a boolean")

        profile = self._profiles[model]
        if not verified:
            return Outcome(None, profile.credits_per_call, 0)

        probability = (
            profile.prior_success(task)
            + self._scenario.success_offsets[user_id][model][task.domain]
        )
        if not 0 <= probability <= 1:
            raise ValueError("synthetic success probability must be in [0, 1]")
        for attempt in range(max_retries + 1):
            if self._draw(task.id, attempt) < probability:
                return Outcome(True, profile.credits_per_call * (attempt + 1), attempt)
        return Outcome(False, profile.credits_per_call * (max_retries + 1), max_retries)
