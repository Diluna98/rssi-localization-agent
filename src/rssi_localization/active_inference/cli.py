"""Command-line entry point for active-inference navigation."""

from __future__ import annotations

import argparse

from .agent import NavigationAgentConfig
from .simulation import run_navigation_episode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a continuous-observation active-inference episode."
    )
    parser.add_argument("--model-size", type=int, default=20)
    parser.add_argument("--goal-resolution", type=int, default=10)
    parser.add_argument("--temporal-horizon", type=int, default=1)
    parser.add_argument("--planning-windows", type=int, default=20)
    parser.add_argument("--policy-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = NavigationAgentConfig(
        model_size=args.model_size,
        goal_resolution=args.goal_resolution,
        temporal_horizon=args.temporal_horizon,
        policy_samples=args.policy_samples,
        random_seed=args.seed,
    )
    result = run_navigation_episode(
        config=config,
        planning_windows=args.planning_windows,
    )
    print(f"initial distance: {result.distances[0]:.3f}")
    print(f"minimum distance: {result.distances.min():.3f}")
    print(f"final distance: {result.distances[-1]:.3f}")
    print(f"moves: {len(result.actions)}")
    print(f"reached goal: {result.reached_goal}")


if __name__ == "__main__":
    main()
