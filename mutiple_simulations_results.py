import numpy as np
from framework import FPA_AuctionEnvironment, SPA_AuctionEnvironment, EpsilonGreedy
import random

NUM_SIMULATIONS = 100
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
AUCTION_CLASS = FPA_AuctionEnvironment  # or SPA_AuctionEnvironment

summary_results = []

for sim_id in range(NUM_SIMULATIONS):
    bidders = [
        EpsilonGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101),
        EpsilonGreedy(name="Agent2", value=0.5, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101)
    ]
    env = AUCTION_CLASS(bidders)

    convergence_count = {bidder: 0 for bidder in bidders}
    last_best_action = {bidder: np.argmax(bidder.q_values) for bidder in bidders}
    converged = False

    for round_idx in range(MAX_ROUNDS):
        bids = {bidder: bidder.place_bid() for bidder in bidders}
        winner, winning_bid, rewards = env._compute_rewards(bids)

        for bidder in bidders:
            bidder.update_strategy(bids[bidder], rewards[bidder])
            best_action = np.argmax(bidder.q_values)

            if best_action == last_best_action[bidder]:
                convergence_count[bidder] += 1
            else:
                convergence_count[bidder] = 0

            last_best_action[bidder] = best_action

        if all(c >= CONVERGENCE_LIMIT for c in convergence_count.values()):
            converged = True
            break

    # Record results
    record = {
        "simulation_id": sim_id,
        "rounds_to_convergence": round_idx + 1 if converged else "NA"
    }

    for bidder in bidders:
        best_index = np.argmax(bidder.q_values)
        best_bid = bidder.bid_options[best_index]
        best_q = bidder.q_values[best_index]
        record[f"converged_bid_{bidder.name}"] = best_bid
        record[f"converged_q_{bidder.name}"] = best_q

    summary_results.append(record)

# Save to file
import pandas as pd
df = pd.DataFrame(summary_results)
df.to_csv("FPA_asymmetry2_results.csv", index=False)