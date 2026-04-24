#!/usr/bin/env python3
"""
Directional ADP-lift gate: sweep snake_draft_sim across ranking modes
and draft slots.  For each (season, ranking_mode), run ModelBot at
every draft slot 1..12 and report the distribution of ModelBot's
finish rank among the 12 teams, plus actual-points lift vs the
11 ADPBot mean.

Answers the question: "does the model have directional pre-draft
signal before we invest in Step 4 calibration?"

Usage:
    python scripts/draft_sim_mode_sweep.py --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "snake_draft_sim",
    PROJECT_ROOT / "scripts" / "snake_draft_sim.py",
)
sd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sd
spec.loader.exec_module(sd)


def run_slot_sweep(season: int, ranking: str, csv_path: Path) -> Dict:
    adp = sd.load_adp_board(season)
    scrape_date = adp["scrape_date"].iloc[0]
    projections = sd.load_model_projections(
        csv_path, ranking=ranking, season=season
    )
    board = sd.build_draft_board(adp, projections)

    ranks: List[int] = []
    model_totals: List[float] = []
    adp_totals: List[float] = []
    wins_vs_adp_mean = 0
    for slot in range(1, sd.TEAMS + 1):
        teams = sd.run_draft(board, model_slot=slot)
        summary = sd.summarize(teams)
        ranks.append(summary["model_rank_of_12"])
        model_totals.append(summary["model_actual_total"])
        adp_totals.append(summary["adp_mean_actual_total"])
        if summary["model_actual_total"] > summary["adp_mean_actual_total"]:
            wins_vs_adp_mean += 1
    return {
        "season": season,
        "scrape_date": scrape_date,
        "ranking": ranking,
        "slot_ranks": ranks,
        "mean_rank": round(statistics.mean(ranks), 2),
        "best_rank": min(ranks),
        "worst_rank": max(ranks),
        "mean_model_actual": round(statistics.mean(model_totals), 1),
        "mean_adp_actual": round(statistics.mean(adp_totals), 1),
        "lift_vs_adp_mean": round(
            statistics.mean(model_totals) - statistics.mean(adp_totals), 1
        ),
        "wins_vs_adp_mean_of_12": wins_vs_adp_mean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--modes", nargs="+",
                    default=["season_sum", "week1", "prior_season"])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    results: List[Dict] = []
    for season in args.seasons:
        matches = sorted(
            (PROJECT_ROOT / "data" / "backtest_results").glob(
                f"ts_backtest_{season}_*_predictions.csv"
            )
        )
        if not matches:
            print(f"No predictions CSV for {season}, skipping.",
                  file=sys.stderr)
            continue
        csv_path = matches[-1]
        for mode in args.modes:
            r = run_slot_sweep(season, mode, csv_path)
            results.append(r)
            print(
                f"  {season}  {mode:>12}  "
                f"mean_rank={r['mean_rank']:.2f}  "
                f"best={r['best_rank']}  "
                f"worst={r['worst_rank']}  "
                f"mean_model_pts={r['mean_model_actual']:.1f}  "
                f"mean_adp_pts={r['mean_adp_actual']:.1f}  "
                f"lift={r['lift_vs_adp_mean']:+.1f}  "
                f"beats_adp_mean_in={r['wins_vs_adp_mean_of_12']}/12_slots"
            )
        print()

    print("Legend:")
    print("  mean_rank:  average ModelBot finish rank (1=best, 12=worst)"
          " across the 12 draft slots.  Pure-luck null is 6.5.")
    print("  lift:       mean_model_pts - mean_adp_pts (avg over 12 slots).")
    print("  beats_adp_mean_in: # of draft slots where ModelBot's roster"
          " actual-total exceeds the 11 ADPBot mean.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
