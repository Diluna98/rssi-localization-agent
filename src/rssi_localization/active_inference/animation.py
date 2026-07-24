"""Create a multi-scenario animation of continuous RSSI navigation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .agent import NavigationAgentConfig
from .environment import GridNavigationEnvironment
from .simulation import NavigationEpisodeResult, run_navigation_episode


@dataclass(frozen=True)
class NavigationScenario:
    """A deterministic start/source pair used in the README animation."""

    label: str
    start: tuple[float, float]
    source: tuple[float, float]
    random_seed: int


README_SCENARIOS = (
    NavigationScenario("A", (487.5, 487.5), (287.5, 187.5), 7),
    NavigationScenario("B", (12.5, 487.5), (312.5, 187.5), 7),
    NavigationScenario("C", (12.5, 12.5), (212.5, 312.5), 7),
    NavigationScenario("D", (487.5, 12.5), (187.5, 312.5), 0),
)


def simulate_scenarios(
    scenarios: tuple[NavigationScenario, ...] = README_SCENARIOS,
    *,
    planning_windows: int = 30,
) -> list[NavigationEpisodeResult]:
    """Run the deterministic scenarios displayed in the animation."""

    results = []
    for scenario in scenarios:
        config = NavigationAgentConfig(
            random_seed=scenario.random_seed,
            policy_samples=300,
        )
        environment = GridNavigationEnvironment(
            model_size=config.model_size,
            start=scenario.start,
            goal=scenario.source,
            random_seed=scenario.random_seed,
        )
        results.append(
            run_navigation_episode(
                config=config,
                environment=environment,
                planning_windows=planning_windows,
            )
        )
    return results


def save_navigation_gif(
    output: str | Path,
    *,
    planning_windows: int = 30,
    fps: int = 5,
) -> Path:
    """Render several navigation episodes concurrently as an animated GIF."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = simulate_scenarios(planning_windows=planning_windows)
    frame_count = max(len(result.positions) for result in results)

    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.2), constrained_layout=True)
    figure.suptitle("Continuous RSSI active-inference navigation", fontsize=14)
    artists = []

    for axis, scenario, result in zip(axes.flat, README_SCENARIOS, results, strict=True):
        axis.set_xlim(0.0, 500.0)
        axis.set_ylim(0.0, 500.0)
        axis.set_aspect("equal")
        axis.set_xticks(np.arange(0.0, 501.0, 100.0))
        axis.set_yticks(np.arange(0.0, 501.0, 100.0))
        axis.grid(color="#d9dee7", linewidth=0.6, zorder=0)
        axis.set_title(
            f"Scenario {scenario.label}  •  start {scenario.start}  •  source {scenario.source}",
            fontsize=8,
        )
        axis.scatter(
            *scenario.start,
            marker="D",
            s=34,
            color="#2ca02c",
            edgecolor="white",
            linewidth=0.7,
            label="start",
            zorder=3,
        )
        axis.scatter(
            *scenario.source,
            marker="*",
            s=130,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            label="RSSI source",
            zorder=4,
        )
        (trail,) = axis.plot([], [], color="#1f77b4", linewidth=2.2, zorder=2)
        agent = axis.scatter(
            [],
            [],
            marker="o",
            s=55,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        status = axis.text(
            0.02,
            0.02,
            "",
            transform=axis.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
            zorder=6,
        )
        artists.append((trail, agent, status, result, scenario))

    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)

    def update(frame: int):
        updated = []
        for trail, agent, status, result, scenario in artists:
            index = min(frame, len(result.positions) - 1)
            positions = result.positions[: index + 1]
            trail.set_data(positions[:, 0], positions[:, 1])
            agent.set_offsets(positions[index : index + 1])
            distance = result.distances[index]
            signal = 30.0 * np.exp(-0.01 * distance)
            state = "source reached" if distance <= 18.0 else "searching"
            status.set_text(
                f"step {index:02d}  |  distance {distance:5.1f}  |  RSSI {signal:4.1f}  |  {state}"
            )
            updated.extend((trail, agent, status))
        return updated

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        blit=False,
        repeat=True,
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=95)
    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the multi-scenario continuous RSSI navigation GIF."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/active_inference_navigation.gif"),
    )
    parser.add_argument("--planning-windows", type=int, default=30)
    parser.add_argument("--fps", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = save_navigation_gif(
        args.output,
        planning_windows=args.planning_windows,
        fps=args.fps,
    )
    print(f"saved animation: {output}")


if __name__ == "__main__":
    main()
