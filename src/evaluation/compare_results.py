"""Summarize and persist results from identical evaluation scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.metrics import summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "scenario_outputs"
DEFAULT_INPUT = RESULTS_DIR / "scenario_results.json"
DEFAULT_OUTPUT = RESULTS_DIR / "comparison.json"
DEFAULT_SUMMARY = RESULTS_DIR / "summary.md"


def compare_results(results_by_method: dict[str, list | dict]) -> dict:
  """Return metric summaries for each method in a results payload."""
  methods = results_by_method.get("methods", results_by_method)
  comparison = {}
  for method, results in methods.items():
    if isinstance(results, dict):
      results = results.get("outcomes", results.get("results", []))
    comparison[str(method)] = summarize(list(results))
  return {"methods": comparison}


def write_comparison(comparison: dict, output_path: str | Path = DEFAULT_OUTPUT) -> Path:
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
  return output_path


def write_summary(comparison: dict, output_path: str | Path = DEFAULT_SUMMARY) -> Path:
  lines = [
    "# Routing Evaluation Summary",
    "",
    "| Method | Episodes | Mean | Median | P95 | CVaR95 | Completion | Wet edges/ep | Deep edges/ep | Deep ep rate | Revisits | Mean regret | P95 regret |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for method, summary in comparison.get("methods", {}).items():
    lines.append(
      f"| {method} | {summary['episodes']} | {summary['mean_travel_time']:.3f} "
      f"| {summary['median_travel_time']:.3f} | {summary['p95_travel_time']:.3f} "
      f"| {summary['cvar95_travel_time']:.3f} | {summary['completion_rate']:.3f} "
      f"| {summary['mean_wet_edge_traversals']:.3f} | {summary['mean_deep_edge_traversals']:.3f} "
      f"| {summary['deep_traversal_episode_rate']:.3f} | {summary['mean_total_revisits']:.3f} "
      f"| {summary['mean_regret_vs_oracle']:.3f} | {summary['p95_regret_vs_oracle']:.3f} |"
    )
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return output_path


def main(input_path: str | Path = DEFAULT_INPUT) -> tuple[Path, Path]:
  input_path = Path(input_path)
  payload = json.loads(input_path.read_text(encoding="utf-8"))
  comparison = compare_results(payload)
  return write_comparison(comparison), write_summary(comparison)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", default=str(DEFAULT_INPUT))
  args = parser.parse_args()
  print(*main(args.input), sep="\n")
