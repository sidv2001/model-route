"""Run the seeded synthetic experiment from the command line."""

import argparse
import json
from pathlib import Path

from .evaluation import POLICIES, ExperimentConfig, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare synthetic contextual routing policies")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=20, help="repetitions of each task template")
    parser.add_argument(
        "--personalize",
        action="store_true",
        help="opt the synthetic users into in-memory learning (off by default)",
    )
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--missing-every",
        type=int,
        default=0,
        help="withhold verification for every Nth task; 0 verifies all",
    )
    parser.add_argument("--report", type=Path, help="write summaries and chosen-action logs as JSON")
    args = parser.parse_args()

    try:
        report = evaluate(
            ExperimentConfig(
                seed=args.seed,
                rounds=args.rounds,
                personalize=args.personalize,
                epsilon=args.epsilon,
                max_retries=args.max_retries,
                missing_every=args.missing_every,
            )
        )
    except ValueError as error:
        parser.error(str(error))

    print("Synthetic offline experiment; credits are not dollars, verification is simulated.")
    print(f"Seed: {args.seed} | Personalization opt-in: {args.personalize}")
    print(f"{'policy':<17} {'verified':>10} {'success':>9} {'credits':>10} {'retries':>8} {'mean reward':>12}")
    for policy in POLICIES:
        summary = report.summaries[policy]
        rate = (
            f"{summary.verified_success_rate:.1%}"
            if summary.verified_success_rate is not None
            else "n/a"
        )
        reward = (
            f"{summary.mean_verified_reward:.3f}"
            if summary.mean_verified_reward is not None
            else "n/a"
        )
        print(
            f"{policy:<17} {summary.verified_count:>4}/{summary.decisions:<5} "
            f"{rate:>9} {summary.total_credits:>10.1f} "
            f"{summary.total_retries:>8} {reward:>12}"
        )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
