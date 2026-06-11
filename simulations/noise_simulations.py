import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import numpy as np
from src import FPA_AuctionEnvironment, SPA_AuctionEnvironment, QlearningGreedy

NUM_SIMULATIONS = 10
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
AUCTION_CLASS = SPA_AuctionEnvironment  # FPA_Auction_Environment or SPA_AuctionEnvironment

summary_results = []

print("Running for noise = 0.01 and value = 1 (symmetric)")

for sim_id in range(NUM_SIMULATIONS):
    bidders = [
        QlearningGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101, delta=0),
        QlearningGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101, delta=0),
    ]
    env = AUCTION_CLASS(bidders)

    convergence_count = {bidder: 0 for bidder in bidders}
    last_best_action = {bidder: np.argmax(bidder.q_values) for bidder in bidders}
    converged = False

    for round_idx in range(MAX_ROUNDS):
        bids = {bidder: bidder.place_bid() for bidder in bidders}

        # No noise is added in the FPA case 
        if isinstance(env, FPA_AuctionEnvironment):
            max_bid = max(bids.values())
            potential_winners = [b for b, amt in bids.items() if amt == max_bid]
            winner = random.choice(potential_winners)
            winning_bid = bids[winner]

            rewards = {bidder: 0.0 for bidder in bids}

            # Use realized_value revealed in begin_round() if available
            w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
            rewards[winner] = float(w_value) - float(winning_bid)
        
        
        else:
            # SPA: second-price; 5% chance a unique top bidder is disqualified
            max_bid = max(bids.values())
            top_candidates = [b for b, amt in bids.items() if amt == max_bid]

            if len(top_candidates) > 1:
                # Tie at the top: pick winner among ties; price = highest other bid
                winner = random.choice(top_candidates)
                second_price = max(amt for bb, amt in bids.items() if bb is not winner)
            else:
                # Unique top
                top = top_candidates[0]
                if random.random() < 0.01:
                    # Disqualify the unique top; choose winner among remaining
                    remaining = [b for b in bids if b is not top]
                    rem_max = max(bids[b] for b in remaining)
                    rem_candidates = [b for b in remaining if bids[b] == rem_max]
                    winner = random.choice(rem_candidates)
                    # Second price = highest losing bid among the remaining (exclude disqualified + new winner)
                    second_price = 0.4
                    #losing_bids = [bids[b] for b in remaining if b is not winner]
                    #second_price = max(losing_bids) if losing_bids else 0.0
                else:
                    # No disqualification
                    winner = top
                    second_price = max(amt for bb, amt in bids.items() if bb is not winner)

            winning_bid = bids[winner]
            rewards = {b: 0.0 for b in bids}
            w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
            rewards[winner] = float(w_value) - float(second_price)

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
df.to_csv("SPA_value_1_noisy.csv", index=False)