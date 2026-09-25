"""On-policy simulation and transparent, non-causal baseline comparisons."""

from dataclasses import asdict, dataclass
from random import Random

from .core import LoggedDecision, Task, log_decision
from .fixtures import load_profiles, load_scenario
from .router import ContextualRouter
from .simulator import SyntheticEnvironment

POLICIES = ("router", "always-quick", "always-careful", "random", "static")
STATIC_BY_COMPLEXITY = {1: "quick", 2: "balanced", 3: "careful"}
MODEL_ORDER = tuple(STATIC_BY_COMPLEXITY.values())


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    seed: int = 7
    rounds: int = 20
    personalize: bool = False
    epsilon: float = 0.15
    max_retries: int = 1
    missing_every: int = 0
    prior_strength: float = 4.0
    cost_weight: float = 0.06
    retry_weight: float = 0.03

    def __post_init__(self) -> None:
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("rounds must be a positive integer")
        if type(self.missing_every) is not int or self.missing_every < 0:
            raise ValueError("missing_every must be a nonnegative integer")
        if type(self.personalize) is not bool:
            raise ValueError("personalize must be a boolean")


@dataclass(frozen=True, slots=True)
class PolicySummary:
    decisions: int
    verified_count: int
    verified_successes: int
    verified_success_rate: float | None
    total_credits: float
    credits_per_task: float
    total_retries: int
    mean_verified_reward: float | None

    @classmethod
    def from_logs(cls, logs: tuple[LoggedDecision, ...]) -> "PolicySummary":
        if not logs:
            raise ValueError("cannot summarize an empty run")
        rewards = [entry.reward for entry in logs if entry.reward is not None]
        verified_count = len(rewards)
        successes = sum(entry.verified_success is True for entry in logs)
        total_credits = sum(entry.total_credits for entry in logs)
        return cls(
            decisions=len(logs),
            verified_count=verified_count,
            verified_successes=successes,
            verified_success_rate=successes / verified_count if verified_count else None,
            total_credits=total_credits,
            credits_per_task=total_credits / len(logs),
            total_retries=sum(entry.retries for entry in logs),
            mean_verified_reward=sum(rewards) / verified_count if verified_count else None,
        )


@dataclass(frozen=True, slots=True)
class Report:
    config: ExperimentConfig
    summaries: dict[str, PolicySummary]
    logs: dict[str, tuple[LoggedDecision, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "synthetic": True,
            "config": asdict(self.config),
            "summaries": {
                policy: asdict(summary) for policy, summary in self.summaries.items()
            },
            "decisions": [
                {"policy": policy, **asdict(entry)}
                for policy, entries in self.logs.items()
                for entry in entries
            ],
        }


def _baseline_choice(policy: str, task: Task, rng: Random) -> tuple[str, float]:
    if policy == "always-quick":
        return "quick", 1.0
    if policy == "always-careful":
        return "careful", 1.0
    if policy == "static":
        return STATIC_BY_COMPLEXITY[task.complexity], 1.0
    if policy == "random":
        return rng.choice(MODEL_ORDER), 1 / len(MODEL_ORDER)
    raise ValueError(f"unknown baseline: {policy}")


def evaluate(config: ExperimentConfig = ExperimentConfig()) -> Report:
    profiles = load_profiles()
    if {profile.name for profile in profiles} != set(MODEL_ORDER):
        raise ValueError("baselines require quick, balanced, and careful profiles")
    scenario = load_scenario(profiles)
    environment = SyntheticEnvironment(profiles, scenario, seed=config.seed)
    tasks = tuple(scenario.task_stream(config.rounds))
    router = ContextualRouter(
        profiles,
        epsilon=config.epsilon,
        prior_strength=config.prior_strength,
        cost_weight=config.cost_weight,
        retry_weight=config.retry_weight,
        max_retries=config.max_retries,
        seed=config.seed,
    )
    if config.personalize:
        for user_id in scenario.success_offsets:
            router.opt_in(user_id)

    logs = {}
    for policy in POLICIES:
        rng = Random(config.seed + 1)
        entries = []
        for index, (user_id, task) in enumerate(tasks):
            if policy == "router":
                decision = router.select(task, user_id)
                model, propensity = decision.model, decision.propensity
            else:
                model, propensity = _baseline_choice(policy, task, rng)
            outcome = environment.run(
                task,
                user_id,
                model,
                max_retries=config.max_retries,
                verified=not (
                    config.missing_every and (index + 1) % config.missing_every == 0
                ),
            )
            entry = (
                router.observe(decision, outcome)
                if policy == "router"
                else log_decision(
                    task, model, propensity, outcome, config.cost_weight, config.retry_weight
                )
            )
            entries.append(entry)
        logs[policy] = tuple(entries)

    return Report(
        config=config,
        summaries={policy: PolicySummary.from_logs(entries) for policy, entries in logs.items()},
        logs=logs,
    )
