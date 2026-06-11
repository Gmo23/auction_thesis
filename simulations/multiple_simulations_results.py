"""
Run many independent simulations of a two-agent auction with stochastic
private values (EpsilonGreedyMultiQ, delta=0.4) and collect per-run
convergence outcomes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src import (
    FPA_AuctionEnvironment,
    SPA_AuctionEnvironment,
    EpsilonGreedyMultiQ,
)

NUM_SIMULATIONS = 100
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
AUCTION_CLASS = SPA_AuctionEnvironment

OUTFILE = None  # set to a path to save


def run():
    summary_results = []

    for sim_id in range(NUM_SIMULATIONS):
        bidders = [
            EpsilonGreedyMultiQ(name="Agent1", value=0.5, a=0.025, b=0.0002,
                                alpha=0.05, gamma=0.99, init_param=101, delta=0.4),
            EpsilonGreedyMultiQ(name="Agent2", value=0.5, a=0.025, b=0.0002,
                                alpha=0.05, gamma=0.99, init_param=101, delta=0.4),
        ]
        env = AUCTION_CLASS(bidders)
        env.run_auction(max_rounds=MAX_ROUNDS, convergence_limit=CONVERGENCE_LIMIT)

        rounds_played = len(env.history)
        converged = rounds_played < MAX_ROUNDS

        record = {
            "simulation_id": sim_id,
            "rounds_to_convergence": rounds_played if converged else "NA",
        }
        for bidder in bidders:
            best_index = int(np.argmax(bidder.q_values))
            record[f"converged_bid_{bidder.name}"] = float(bidder.bid_options[best_index])
            record[f"converged_q_{bidder.name}"] = float(bidder.q_values[best_index])

        summary_results.append(record)

    df = pd.DataFrame(summary_results)
    if OUTFILE:
        df.to_csv(OUTFILE, index=False)
        print(f"Saved {len(df)} rows to {OUTFILE}")
    return df


if __name__ == "__main__":
    run()
