"""Auction environments: abstract base + First-Price and Second-Price implementations."""

import inspect
import random
import numpy as np
from abc import ABC, abstractmethod


def _best_action_signature(bidder):
    """
    Summarize a bidder's current 'best action(s)' for convergence checks.

    - MultiQ bidders: tuple of argmax indices over states (-1, 0, +1).
    - LinUCB: ("linucb", tuple_of_top_indices).
    - 1D q_values: ("q1", tuple_of_top_indices).
    - 2D q_values: ("q2", tuple_of_per-column argmax).
    """
    atol = 1e-12

    if hasattr(bidder, "q_values_map"):
        cols = [-1, 0, +1]
        return tuple(int(np.argmax(bidder.q_values_map[c])) for c in cols)

    if hasattr(bidder, "A") and hasattr(bidder, "b") and hasattr(bidder, "feature_fn") \
       and hasattr(bidder, "alpha") and hasattr(bidder, "bid_options"):
        scores = []
        for i, bid in enumerate(bidder.bid_options):
            x = bidder.feature_fn(bidder, float(bid))
            theta = np.linalg.solve(bidder.A[i], bidder.b[i])
            mean = float(x @ theta)
            Ainv_x = np.linalg.solve(bidder.A[i], x)
            bonus = float(bidder.alpha * np.sqrt(x @ Ainv_x))
            scores.append(mean + bonus)
        scores = np.asarray(scores, dtype=float)
        mx = scores.max()
        top = tuple(np.flatnonzero(np.isclose(scores, mx, atol=atol)))
        return ("linucb", top)

    q = getattr(bidder, "q_values", None)
    if isinstance(q, np.ndarray):
        if q.ndim == 1:
            mx = q.max()
            top = tuple(np.flatnonzero(np.isclose(q, mx, atol=atol)))
            return ("q1", top)
        elif q.ndim == 2:
            top = tuple(int(np.argmax(q[:, j])) for j in range(q.shape[1]))
            return ("q2", top)

    return ("unknown", tuple())


def _needs_next_action(agent):
    return getattr(agent, "requires_next_action", False) or \
           ("next_bid" in inspect.signature(agent.update_strategy).parameters)


class AbstractAuctionEnvironment(ABC):
    """Abstract base class for a repeated auction environment."""

    def __init__(self, bidders):
        self.bidders = bidders
        self.history = []

    @abstractmethod
    def _compute_rewards(self, bids):
        """
        Given a dictionary of {bidder: bid}, return (winner, winning_bid, rewards_dict).
        """
        pass

    def run_auction(self, max_rounds=1000000, convergence_limit=1000):
        """Run repeated auction rounds, stopping early on convergence."""
        convergence_count = {bidder: 0 for bidder in self.bidders}
        last_best_sig = {bidder: _best_action_signature(bidder) for bidder in self.bidders}
        queued_next = {}

        for round_index in range(max_rounds):
            for bidder in self.bidders:
                if hasattr(bidder, "begin_round"):
                    bidder.begin_round()

            bids = {}
            for bidder in self.bidders:
                if bidder in queued_next:
                    bids[bidder] = queued_next.pop(bidder)
                else:
                    bids[bidder] = bidder.place_bid()

            winner, winning_bid, rewards = self._compute_rewards(bids)

            next_bid = {}
            for bidder in self.bidders:
                if _needs_next_action(bidder):
                    nb = bidder.place_bid()
                    next_bid[bidder] = nb
                    queued_next[bidder] = nb

            for bidder in self.bidders:
                if _needs_next_action(bidder):
                    bidder.update_strategy(bids[bidder], rewards[bidder],
                                           next_bid=next_bid[bidder], terminal=False)
                else:
                    bidder.update_strategy(bids[bidder], rewards[bidder])

                current_sig = _best_action_signature(bidder)
                if current_sig == last_best_sig[bidder]:
                    convergence_count[bidder] += 1
                else:
                    convergence_count[bidder] = 0
                    last_best_sig[bidder] = current_sig

            q_snapshot = {}
            for bidder in self.bidders:
                qv = getattr(bidder, "q_values", None)
                if isinstance(qv, np.ndarray):
                    q_snapshot[bidder.name] = qv.copy()
                else:
                    q_snapshot[bidder.name] = qv

            self.history.append((bids, winner.name, winning_bid, rewards.copy(), q_snapshot))

            if all(c >= convergence_limit for c in convergence_count.values()):
                print(f"Convergence detected after {round_index + 1} rounds.")
                for bidder in self.bidders:
                    sig = last_best_sig[bidder]
                    kind, bid = sig

                    if kind == "linucb":
                        idx = int(bid[0])
                        bid_val = float(bidder.bid_options[idx])
                        print(f"Bidder {bidder.name} converged to {bid_val:.2f}")
                    elif kind in ("q1", "q?", "q2") and hasattr(bidder, "bid_options"):
                        if isinstance(bid, tuple) and len(bid) == 1:
                            idx = int(bid[0])
                            bid_val = float(bidder.bid_options[idx])
                            print(f"Bidder {bidder.name} converged to {bid_val:.2f}")
                        else:
                            print(f"Bidder {bidder.name} signature {bid} (tie)")
                    else:
                        print(f"Bidder {bidder.name} converged to {(bid + 1) * 0.05}")
                break

            if round_index + 1 == max_rounds:
                print("Agents have not converged after 1,000,000 rounds")


class FPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Repeated First-Price Auction."""

    def _compute_rewards(self, bids):
        max_bid = max(bids.values())
        potential_winners = [b for b, amt in bids.items() if amt == max_bid]
        winner = random.choice(potential_winners)
        winning_bid = bids[winner]

        rewards = {bidder: 0.0 for bidder in bids}
        w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
        rewards[winner] = float(w_value) - float(winning_bid)
        return winner, winning_bid, rewards


class SPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Repeated Second-Price Auction."""

    def _compute_rewards(self, bids):
        max_bid = max(bids.values())
        potential_winners = [b for b, amt in bids.items() if amt == max_bid]
        winner = random.choice(potential_winners)

        other_bids = [amt for b, amt in bids.items() if b != winner]
        second_price = max(other_bids) if other_bids else 0.0

        rewards = {bidder: 0.0 for bidder in bids}
        w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
        rewards[winner] = float(w_value) - float(second_price)
        return winner, max_bid, rewards
