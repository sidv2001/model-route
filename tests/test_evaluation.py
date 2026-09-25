import json

import pytest

from model_route import Task
from model_route.evaluation import ExperimentConfig, evaluate
from model_route.fixtures import load_profiles, load_scenario
from model_route.simulator import SyntheticEnvironment


def test_seeded_replay_and_chosen_only_report_shape() -> None:
    config = ExperimentConfig(seed=17, rounds=2, personalize=True, missing_every=5)
    report = evaluate(config)
    data = report.as_dict()

    assert data == evaluate(config).as_dict()
    assert set(data) == {"synthetic", "config", "summaries", "decisions"}
    assert data["synthetic"] is True
    assert set(data["summaries"]) == {
        "router",
        "always-quick",
        "always-careful",
        "random",
        "static",
    }
    assert len(data["decisions"]) == 5 * 2 * 9 * 2
    assert set(data["decisions"][0]) == {
        "policy",
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
    assert "demo-01" not in json.dumps(data)
    assert "demo-02" not in json.dumps(data)
    for summary in report.summaries.values():
        assert summary.decisions == 36
        assert summary.verified_count == 29
        assert 0 <= summary.verified_successes <= summary.verified_count


def test_cost_and_retry_accounting_includes_failed_and_unverified_calls() -> None:
    report = evaluate(ExperimentConfig(rounds=2, missing_every=4, max_retries=2))
    credits_by_model = {profile.name: profile.credits_per_call for profile in load_profiles()}

    for policy, logs in report.logs.items():
        for entry in logs:
            assert 0 <= entry.retries <= 2
            assert entry.total_credits == pytest.approx(
                credits_by_model[entry.model] * (entry.retries + 1)
            )
            if entry.verified_success is None:
                assert entry.reward is None
                assert entry.retries == 0
            else:
                assert entry.reward == pytest.approx(
                    float(entry.verified_success)
                    - 0.06 * entry.total_credits
                    - 0.03 * entry.retries
                )
        assert report.summaries[policy].total_credits == pytest.approx(
            sum(entry.total_credits for entry in logs)
        )
        assert report.summaries[policy].total_retries == sum(entry.retries for entry in logs)


def test_baseline_and_router_logged_propensities() -> None:
    report = evaluate(ExperimentConfig(rounds=1, epsilon=0.2))
    for policy, entries in report.logs.items():
        for entry in entries:
            if policy == "router":
                assert entry.propensity == pytest.approx(0.2 / 3) or entry.propensity == (
                    pytest.approx(0.8 + 0.2 / 3)
                )
            elif policy == "random":
                assert entry.propensity == pytest.approx(1 / 3)
            else:
                assert entry.propensity == 1
            if policy == "always-quick":
                assert entry.model == "quick"
            if policy == "always-careful":
                assert entry.model == "careful"
            if policy == "static":
                assert entry.model == {
                    1: "quick",
                    2: "balanced",
                    3: "careful",
                }[entry.complexity]


def test_no_verified_feedback_has_no_success_or_reward_metric() -> None:
    report = evaluate(ExperimentConfig(rounds=1, personalize=True, missing_every=1))
    for summary in report.summaries.values():
        assert summary.decisions == 18
        assert summary.verified_count == 0
        assert summary.verified_success_rate is None
        assert summary.mean_verified_reward is None
        assert summary.total_credits > 0
        assert summary.total_retries == 0
    assert all(
        entry.verified_success is None
        for logs in report.logs.values()
        for entry in logs
    )


def test_simulator_only_runs_selected_model_with_bounded_retries() -> None:
    profiles = load_profiles()
    environment = SyntheticEnvironment(profiles, load_scenario(profiles), seed=7)
    outcomes = [
        environment.run(
            Task(f"case-{index}", "code", 3),
            "demo-02",
            "quick",
            max_retries=2,
        )
        for index in range(12)
    ]

    assert any(outcome.retries == 2 for outcome in outcomes)
    assert all(outcome.total_credits == outcome.retries + 1 for outcome in outcomes)
    assert environment.run(
        Task("unverified", "code", 3), "demo-02", "quick", max_retries=2, verified=False
    ).verified_success is None
    with pytest.raises(ValueError, match="max_retries"):
        environment.run(Task("bad", "code", 1), "demo-02", "quick", max_retries=3)
