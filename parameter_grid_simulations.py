import numpy as np
import pandas as pd
from framework import FPA_AuctionEnvironment, SPA_AuctionEnvironment, EpsilonGreedy
import random

GAMMA_GRID = [0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
ALPHA_GRID = [0.01, 0.05, 0.1, 0.25, 0.5, 1]  # order matters (for ID mapping)
BETA_GRID = [0, 0.0001, 0.00015, 0.0002, 0.00025, 0.0003, 0.0005, 0.001]
NUM_TRIALS_PER_COMBO = 100
MAX_ROUNDS = 1_000_000
CONVERGENCE_LIMIT = 1000
AUCTION_CLASS = SPA_AuctionEnvironment  # or SPA_AuctionEnvironment


def gamma_grid(GAMMA_GRID):
    summary_results = []

    n = len(GAMMA_GRID)
    total_combos = n * (n + 1) // 2  # upper triangle incl. diagonal
    combo_counter = 0

    for i, gamma1 in enumerate(GAMMA_GRID):          # i = 0..4 (Agent1's gamma index)
        for j in range(i, n):      
            gamma2 = GAMMA_GRID[j]

            grid_id = int(f"{i+1}{j+1}")  # e.g., i=0,j=0 -> 11; i=1,j=2 -> 23

            for trial in range(NUM_TRIALS_PER_COMBO):
                # (Optional) vary seeds per trial to diversify randomness while keeping it reproducible
                seed_val = (i * 10000) + (j * 1000) + trial
                random.seed(seed_val)
                np.random.seed(seed_val)

                #Ensures that the intialisations are optimistic for different gamma
                init_param1 = 1.0 / (1.0 - gamma1)
                init_param2 = 1.0 / (1.0 - gamma2)

                bidders = [
                    EpsilonGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha=0.05,
                                gamma=gamma1, init_param=init_param1, delta=0),
                    EpsilonGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha=0.05,
                                gamma=gamma2, init_param=init_param2, delta=0),
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

                # Record results for this trial (shares the same grid_id with the other 99 in the combo)
                record = {
                    "grid_id": grid_id,
                    "trial": trial,  
                    "gamma_agent1": gamma1,
                    "gamma_agent2": gamma2,
                    "rounds_to_convergence": (round_idx + 1) if converged else "NA",
                }

                for bidder in bidders:
                    best_index = np.argmax(bidder.q_values)
                    best_bid = bidder.bid_options[best_index]
                    best_q = bidder.q_values[best_index]
                    record[f"converged_bid_{bidder.name}"] = best_bid
                    record[f"converged_q_{bidder.name}"] = best_q

                summary_results.append(record)

            combo_counter += 1
            print(f"Completed combo {combo_counter}/{total_combos} "
                  f"grid_id={grid_id} (Agent1 alpha={gamma1}, Agent2 alpha={gamma2})")
            

    # Save all 25×100 rows into one CSV
    df = pd.DataFrame(summary_results)

    # Saving
    outfile = "parameter_grid_results_SPA_alpha.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")

def alpha_grid(ALPHA_GRID):
    summary_results = []

    n = len(ALPHA_GRID)
    total_combos = n * (n + 1) // 2  # upper triangle incl. diagonal
    combo_counter = 0

    for i, alpha1 in enumerate(ALPHA_GRID):
        for j in range(i, n):  # <-- only j >= i
            alpha2 = ALPHA_GRID[j]

            # Grid ID: first digit = i+1 (Agent1), second digit = j+1 (Agent2)
            grid_id = int(f"{i+1}{j+1}")

            for trial in range(NUM_TRIALS_PER_COMBO):
                # seed per (i,j,trial) for reproducibility without duplication
                seed_val = (i * 10000) + (j * 1000) + trial
                random.seed(seed_val)
                np.random.seed(seed_val)

                bidders = [
                    EpsilonGreedy(
                        name="Agent1", value=1, a=0.025, b=0.0002, alpha=alpha1,
                        gamma=0.99, init_param=1.0/(1.0-0.99), delta=0
                    ),
                    EpsilonGreedy(
                        name="Agent2", value=1, a=0.025, b=0.0002, alpha=alpha2,
                        gamma=0.99, init_param=1.0/(1.0-0.99), delta=0
                    ),
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

                # Record one trial
                record = {
                    "grid_id": grid_id,
                    "trial": trial,
                    "alpha_agent1": alpha1,
                    "alpha_agent2": alpha2,
                    "rounds_to_convergence": (round_idx + 1) if converged else "NA",
                }
                for bidder in bidders:
                    best_index = np.argmax(bidder.q_values)
                    best_bid = bidder.bid_options[best_index]
                    best_q = bidder.q_values[best_index]
                    record[f"converged_bid_{bidder.name}"] = best_bid
                    record[f"converged_q_{bidder.name}"] = best_q

                summary_results.append(record)

            combo_counter += 1
            print(f"Completed combo {combo_counter}/{total_combos} "
                  f"grid_id={grid_id} (Agent1 alpha={alpha1}, Agent2 alpha={alpha2})")

    # Save all rows into one CSV
    df = pd.DataFrame(summary_results)
    outfile = "parameter_grid_results_FPA_alpha.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")

def beta_grid(BETA_GRID):
    summary_results = []

    n = len(BETA_GRID)
    total_combos = n * (n + 1) // 2  # upper triangle incl. diagonal
    combo_counter = 0

    for i, beta1 in enumerate(BETA_GRID):
        for j in range(i, n):  # <-- only j >= i
            beta2 = BETA_GRID[j]

            # Grid ID: first digit = i+1 (Agent1), second digit = j+1 (Agent2)
            grid_id = int(f"{i+1}{j+1}")

            for trial in range(NUM_TRIALS_PER_COMBO):
                # seed per (i,j,trial) for reproducibility without duplication
                seed_val = (i * 10000) + (j * 1000) + trial
                random.seed(seed_val)
                np.random.seed(seed_val)

                bidders = [
                    EpsilonGreedy(
                        name="Agent1", value=1, a=0.025, b=beta1, alpha=0.05,
                        gamma=0.99, init_param=1.0/(1.0-0.99), delta=0
                    ),
                    EpsilonGreedy(
                        name="Agent2", value=1, a=0.025, b=beta2, alpha=0.05,
                        gamma=0.99, init_param=1.0/(1.0-0.99), delta=0
                    ),
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

                # Record one trial
                record = {
                    "grid_id": grid_id,
                    "trial": trial,
                    "beta_agent1": beta1,
                    "beta_agent2": beta2,
                    "rounds_to_convergence": (round_idx + 1) if converged else "NA",
                }
                for bidder in bidders:
                    best_index = np.argmax(bidder.q_values)
                    best_bid = bidder.bid_options[best_index]
                    best_q = bidder.q_values[best_index]
                    record[f"converged_bid_{bidder.name}"] = best_bid
                    record[f"converged_q_{bidder.name}"] = best_q

                summary_results.append(record)

            combo_counter += 1
            print(f"Completed combo {combo_counter}/{total_combos} "
                  f"grid_id={grid_id} (Agent1 alpha={beta1}, Agent2 alpha={beta2})")

    # Save all rows into one CSV
    df = pd.DataFrame(summary_results)
    outfile = "parameter_grid_results_FPA_beta.csv"
    df.to_csv(outfile, index=False)
    print(f"Saved {len(df)} rows to {outfile}")

alpha_grid(ALPHA_GRID)