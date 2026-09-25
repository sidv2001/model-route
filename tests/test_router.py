from dataclasses import asdict, replace

import pytest

from model_route import ContextualRouter, Outcome, Task
from model_route.fixtures import load_profiles


def test_cold_start_uses_capability_cost_and_complexity_priors() -> None:
    router = ContextualRouter(load_profiles(), epsilon=0)
    easy = Task("easy", "code", 1)
    hard = Task("hard", "code", 3)

    expected_calls = 1 + (1 - 0.76)
    expected_success = 1 - (1 - 0.76) ** 2
    assert router.estimate(easy, "quick") == pytest.approx(
        expected_success - 0.06 * expected_calls - 0.03 * (expected_calls - 1)
    )
    assert router.select(easy).model == "quick"
    assert router.select(hard).model == "careful"


def test_exact_epsilon_greedy_propensities_and_chosen_action_only() -> None:
    router = ContextualRouter(load_profiles(), epsilon=0.3, seed=12)
    task = Task("propensity", "code", 1)
    probabilities = router.action_probabilities(task)

    assert probabilities == pytest.approx(
        {"quick": 0.8, "balanced": 0.1, "careful": 0.1}
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)
    decision = router.select(task)
    assert decision.propensity == pytest.approx(probabilities[decision.model])

    with pytest.raises(ValueError, match="unknown or already observed"):
        router.observe(
            replace(decision, model="careful" if decision.model != "careful" else "quick"),
            Outcome(True, 1, 0),
        )
    entry = router.observe(decision, Outcome(True, 1, 0))
    assert entry.model == decision.model
    assert entry.propensity == decision.propensity
    assert set(asdict(entry)) == {
        "task_id",
        "domain",
        "complexity",
        "model",
        "propensity",
        "verified_success",
        "total_credits",
        "retries",
        "reward",
    }
    with pytest.raises(ValueError, match="unknown or already observed"):
        router.observe(decision, Outcome(True, 1, 0))


def test_per_user_feedback_changes_only_matching_context() -> None:
    router = ContextualRouter(load_profiles(), epsilon=0, prior_strength=0.5)
    router.opt_in("user-a")
    router.opt_in("user-b")
    code = Task("one", "code", 1)
    writing = Task("two", "writing", 1)
    other_complexity = Task("three", "code", 2)
    unchanged_writing = router.estimate(writing, "quick", "user-a")
    unchanged_complexity = router.estimate(other_complexity, "quick", "user-a")

    decision = router.select(code, "user-a")
    assert decision.model == "quick"
    router.observe(decision, Outcome(False, total_credits=2, retries=1))

    assert router.select(Task("four", "code", 1), "user-a").model == "balanced"
    assert router.select(Task("five", "code", 1), "user-b").model == "quick"
    assert router.estimate(writing, "quick", "user-a") == unchanged_writing
    assert router.estimate(other_complexity, "quick", "user-a") == unchanged_complexity
    assert len(router.history_for("user-a")) == 1
    assert router.history_for("user-b") == ()


def test_opt_out_forget_and_in_flight_decision_remain_anonymous() -> None:
    router = ContextualRouter(load_profiles(), epsilon=0)
    task = Task("anonymous-task", "code", 1)
    unconsented = router.select(task, "private-user")
    unconsented_log = router.observe(unconsented, Outcome(True, 1, 0))
    assert router.history_for("private-user") == ()
    assert "private-user" not in str(asdict(unconsented))
    assert "private-user" not in str(asdict(unconsented_log))

    router.opt_in("private-user")
    pending = router.select(task, "private-user")
    router.forget("private-user")
    router.observe(pending, Outcome(False, 2, 1))
    assert router.history_for("private-user") == ()
    router.opt_in("private-user")
    assert router.estimate(task, "quick", "private-user") == router.estimate(task, "quick")
    assert router.select(task, "private-user").model == "quick"


def test_missing_feedback_counts_cost_without_training() -> None:
    router = ContextualRouter(load_profiles(), epsilon=0)
    router.opt_in("user-a")
    task = Task("unverified", "code", 1)
    initial = router.estimate(task, "quick", "user-a")
    decision = router.select(task, "user-a")
    entry = router.observe(decision, Outcome(None, total_credits=1, retries=0))

    assert entry.verified_success is None
    assert entry.reward is None
    assert entry.total_credits == 1
    assert router.history_for("user-a") == ()
    assert router.estimate(task, "quick", "user-a") == initial


@pytest.mark.parametrize(
    "task",
    [Task("low", "code", 1), Task("mid", "writing", 2), Task("high", "reasoning", 3)],
)
def test_action_probabilities_normalize_at_exploration_extremes(task: Task) -> None:
    for epsilon in (0.0, 1.0):
        router = ContextualRouter(load_profiles(), epsilon=epsilon)
        probabilities = router.action_probabilities(task)
        assert sum(probabilities.values()) == pytest.approx(1)
        assert all(0 <= probability <= 1 for probability in probabilities.values())
        if epsilon == 1:
            assert all(
                probability == pytest.approx(1 / 3)
                for probability in probabilities.values()
            )
