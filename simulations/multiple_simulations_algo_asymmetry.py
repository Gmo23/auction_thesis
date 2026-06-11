# run_pairs.py
import numpy as np
import pandas as pd
from itertools import product
from typing import Tuple
from itertools import combinations_with_replacement

from src import (
    SPA_AuctionEnvironment,
    FPA_AuctionEnvironment,
    QlearningGreedy,
    SarsaGreedy,
    ContextualLinUCB,
)
from src.environments import _best_action_signature  # tie-aware helper

# ---------------- Config ----------------
NUM_SIMULATIONS   = 100
MAX_ROUNDS        = 1_000_000
CONVERGENCE_LIMIT = 1000
ENV_CLASS         = SPA_AuctionEnvironment   # change to FPA_AuctionEnvironment if needed

# Fixed bid grid assumptions (fits your agents)
N_BIDS = 19
BID_GRID = np.array([i * 0.05 for i in range(1, N_BIDS + 1)], dtype=float)

# Optional reproducibility
GLOBAL_SEED = 12345
np.random.seed(GLOBAL_SEED)

# ------------- Feature function for LinUCB -------------
def default_linucb_features(bidder, bid: float) -> np.ndarray:
    """
    Simple 4D feature map (you can customize):
      [1, realized_value, bid, realized_value - bid]
    """
    rv = getattr(bidder, "realized_value", bidder.value)
    return np.array([1.0, float(rv), float(bid), float(rv) - float(bid)], dtype=float)

# ------------- Agent factory -------------
def make_bidder(alg_name: str, name: str, value: float = 1):
    """
    Create a bidder by algorithm name with sensible defaults matching your framework.
    QLearningGreedy and SarsaGreedy use optimistic starts; LinUCB uses reg=1, alpha=1.
    """
    alg_name = alg_name.lower()
    if alg_name in ("qlearning", "qlearninggreedy", "ql"):
        return QlearningGreedy(
            name=name, value=value,
            a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
            init_param=101, num_bids=N_BIDS
        )
    elif alg_name in ("sarsa", "sarsagreedy"):
        # on-policy SARSA; your environment loop must provide next_bid (already done in Option A)
        return SarsaGreedy(
            name=name, value=value,
            a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
            init_param=101, n_bids=N_BIDS
        )
    elif alg_name in ("contextuallinucb", "lin-ucb"):
        return ContextualLinUCB(
            name=name, value=value,
            d=4, alpha=1.0, reg=1.0, n_bids=N_BIDS,
            feature_fn=default_linucb_features
        )
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

# ------------- Utilities -------------
def _singleton_best_index(bidder):
    kind, payload = _best_action_signature(bidder)
    if isinstance(payload, tuple) and len(payload) == 1:
        return True, int(payload[0])
    return False, -1


def _converged_rounds(env, max_rounds: int) -> Tuple[bool, int]:
    """
    Determine if the run converged early:
      - If length of env.history < max_rounds -> converged
      - rounds_to_convergence = len(env.history) (number of rounds actually played)
    """
    rounds_played = len(env.history)
    converged = rounds_played < max_rounds
    return converged, rounds_played

# ------------- Main experiment -------------
def run_experiment(value_agent1=1, value_agent2=1,
                   include_self_matches=True,
                   out_csv="algorithmic_asymmetry.csv"):
    alg_labels = ["QLearningGreedy", "SarsaGreedy", "ContextualLinUCB"]

    # Unordered pairs (upper triangle). Includes self-matches if requested.
    pairs = list(combinations_with_replacement(alg_labels, 2))
    if not include_self_matches:
        pairs = [(a, b) for (a, b) in pairs if a != b]

    rows = []

    for algA, algB in pairs:
        for sim_id in range(NUM_SIMULATIONS):
            # Fresh agents per simulation
            bidder1 = make_bidder(algA, name="Agent1", value=value_agent1)
            bidder2 = make_bidder(algB, name="Agent2", value=value_agent2)
            bidders = [bidder1, bidder2]

            # Create environment (Option A — environment-driven loop)
            env = ENV_CLASS(bidders)

            # Run until convergence or max rounds
            env.run_auction(max_rounds=MAX_ROUNDS, convergence_limit=CONVERGENCE_LIMIT)

            converged, rounds = _converged_rounds(env, MAX_ROUNDS)

            # Infer converged bids (only if we have a unique best action per bidder)
            a1_single, a1_idx = _singleton_best_index(bidder1)
            a2_single, a2_idx = _singleton_best_index(bidder2)

            rec = {
                "pair_first":  algA,
                "pair_second": algB,
                "simulation_id": sim_id,
                "rounds_to_convergence": rounds if converged else np.nan,
                # Report converged bids when singleton; else NaN (tie/undefined)
                "converged_bid_Agent1": float(BID_GRID[a1_idx]) if a1_single else np.nan,
                "converged_bid_Agent2": float(BID_GRID[a2_idx]) if a2_single else np.nan,
            }

            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")
    return df


if __name__ == "__main__":
    run_experiment(
        value_agent1=1,
        value_agent2=1,
        include_self_matches=True,
        out_csv="SPA_algorithmic_asymmetry.csv",
    )
