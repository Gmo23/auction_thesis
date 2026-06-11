"""
Noisy SPA: with probability `disqualification_prob`, a unique top bidder is
disqualified and the auction is resolved among the remainder. Implemented as
a subclass of SPA_AuctionEnvironment so the framework's standard loop still
drives convergence detection.
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

NUM_SIMULATIONS = 10
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
DISQUALIFICATION_PROB = 0.01
FALLBACK_SECOND_PRICE = 0.4


class NoisySPA(SPA_AuctionEnvironment):
    """SPA where a unique top bidder is disqualified with some probability."""

    def __init__(self, bidders, disqualification_prob=DISQUALIFICATION_PROB,
                 fallback_second_price=FALLBACK_SECOND_PRICE):
        super().__init__(bidders)
        self.disqualification_prob = disqualification_prob
        self.fallback_second_price = fallback_second_price

    def _compute_rewards(self, bids):
        max_bid = max(bids.values())
        top_candidates = [b for b, amt in bids.items() if amt == max_bid]

        if len(top_candidates) > 1:
            winner = random.choice(top_candidates)
            second_price = max(amt for bb, amt in bids.items() if bb is not winner)
        else:
            top = top_candidates[0]
            if random.random() < self.disqualification_prob:
                remaining = [b for b in bids if b is not top]
                rem_max = max(bids[b] for b in remaining)
                rem_candidates = [b for b in remaining if bids[b] == rem_max]
                winner = random.choice(rem_candidates)
                second_price = self.fallback_second_price
            else:
                winner = top
                second_price = max(amt for bb, amt in bids.items() if bb is not winner)

        rewards = {b: 0.0 for b in bids}
        w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
        rewards[winner] = float(w_value) - float(second_price)
        return winner, max_bid, rewards


def run(env_cls=NoisySPA, outfile="SPA_value_1_noisy.csv"):
    print("Running for noise = 0.01 and value = 1 (symmetric)")
    summary_results = []

    for sim_id in range(NUM_SIMULATIONS):
        bidders = [
            QlearningGreedy(name="Agent1", value=1, a=0.025, b=0.0002,
                            alpha=0.05, gamma=0.99, init_param=101, delta=0),
            QlearningGreedy(name="Agent2", value=1, a=0.025, b=0.0002,
                            alpha=0.05, gamma=0.99, init_param=101, delta=0),
        ]
        env = env_cls(bidders)
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
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")
    return df


if __name__ == "__main__":
    run()
