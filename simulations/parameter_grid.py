"""
Sweep two-agent Q-learning over grids of gamma, alpha, or beta and save
convergence outcomes to CSV. Run from repo root:

    python -m simulations.parameter_grid
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
import pandas as pd

from src import (
    FPA_AuctionEnvironment,
    SPA_AuctionEnvironment,
    QlearningGreedy,
)

GAMMA_GRID = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
BETA_GRID = [0, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0005, 0.001]

NUM_TRIALS_PER_COMBO = 100
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
AUCTION_CLASS = SPA_AuctionEnvironment


def _run_one_trial(bidders, seed_val):
    """Seed RNGs, run the auction, return (converged, rounds_played)."""
    random.seed(seed_val)
    np.random.seed(seed_val)

    env = AUCTION_CLASS(bidders)
    env.run_auction(max_rounds=MAX_ROUNDS, convergence_limit=CONVERGENCE_LIMIT)

    rounds_played = len(env.history)
    converged = rounds_played < MAX_ROUNDS
    return converged, rounds_played


def _record_convergence(record, bidders):
    """Add per-bidder converged-bid and converged-Q columns to record."""
    for bidder in bidders:
        best_index = int(np.argmax(bidder.q_values))
        record[f"converged_bid_{bidder.name}"] = float(bidder.bid_options[best_index])
        record[f"converged_q_{bidder.name}"] = float(bidder.q_values[best_index])
    return record


def _sweep(grid, make_bidders, label, outfile):
    """
    Run the upper-triangle sweep (including the diagonal) for a 1D parameter grid.

    make_bidders(p1, p2) -> [bidder1, bidder2]
    label is a short name used for the grid columns (e.g. "gamma", "alpha", "beta").
    """
    n = len(grid)
    total_combos = n * (n + 1) // 2
    combo_counter = 0
    results = []

    for i, p1 in enumerate(grid):
        for j in range(i, n):
            p2 = grid[j]
            grid_id = int(f"{i + 1}{j + 1}")

            for trial in range(NUM_TRIALS_PER_COMBO):
                seed_val = (i * 10000) + (j * 1000) + trial
                bidders = make_bidders(p1, p2)
                converged, rounds_played = _run_one_trial(bidders, seed_val)

                record = {
                    "grid_id": grid_id,
                    "trial": trial,
                    f"{label}_agent1": p1,
                    f"{label}_agent2": p2,
                    "rounds_to_convergence": rounds_played if converged else "NA",
                }
                _record_convergence(record, bidders)
                results.append(record)

            combo_counter += 1
            print(f"Completed combo {combo_counter}/{total_combos} "
                  f"grid_id={grid_id} ({label}1={p1}, {label}2={p2})")

    df = pd.DataFrame(results)
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")
    return df


def gamma_grid(grid=GAMMA_GRID, outfile="parameter_grid_results_SPA_gamma.csv"):
    def make_bidders(g1, g2):
        return [
            QlearningGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha=0.05,
                            gamma=g1, init_param=1.0 / (1.0 - g1), delta=0),
            QlearningGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha=0.05,
                            gamma=g2, init_param=1.0 / (1.0 - g2), delta=0),
        ]
    return _sweep(grid, make_bidders, "gamma", outfile)


def alpha_grid(grid=ALPHA_GRID, outfile="parameter_grid_results_FPA_alpha.csv"):
    init = 1.0 / (1.0 - 0.99)

    def make_bidders(a1, a2):
        return [
            QlearningGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha=a1,
                            gamma=0.99, init_param=init, delta=0),
            QlearningGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha=a2,
                            gamma=0.99, init_param=init, delta=0),
        ]
    return _sweep(grid, make_bidders, "alpha", outfile)


def beta_grid(grid=BETA_GRID, outfile="parameter_grid_results_FPA_beta.csv"):
    init = 1.0 / (1.0 - 0.99)

    def make_bidders(b1, b2):
        return [
            QlearningGreedy(name="Agent1", value=1, a=0.025, b=b1, alpha=0.05,
                            gamma=0.99, init_param=init, delta=0),
            QlearningGreedy(name="Agent2", value=1, a=0.025, b=b2, alpha=0.05,
                            gamma=0.99, init_param=init, delta=0),
        ]
    return _sweep(grid, make_bidders, "beta", outfile)


if __name__ == "__main__":
    alpha_grid()
