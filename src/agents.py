"""Bidder agents: Q-learning, SARSA, LinUCB, and multi-Q variants."""

import random
import numpy as np


class QlearningGreedy:
    """ε-greedy Q-learning agent over a discrete bid grid."""

    def __init__(self, name, value, a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
                 init_param=101, delta=0, num_bids=19):
        self.name = name
        self.value = value
        self.a = a              # exploration constant
        self.b = b              # epsilon decay rate
        self.time_step = 0
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.number_of_bids = num_bids
        self.init_param = init_param
        self.delta = delta      # stochastic value shock (unused in this class)

        self.bid_options = np.array([i * 0.05 for i in range(1, self.number_of_bids + 1)])
        self.q_values = np.full(self.number_of_bids, float(self.init_param))

    def place_bid(self):
        epsilon_t = self.a * np.exp(-self.b * self.time_step)

        if np.random.rand() < epsilon_t:
            action = np.random.randint(self.number_of_bids)
        else:
            max_value = np.max(self.q_values)
            max_indices = np.flatnonzero(self.q_values == max_value)
            action = np.random.choice(max_indices)

        self.time_step += 1
        return self.bid_options[action]

    def update_strategy(self, bid, reward):
        action = np.where(self.bid_options == bid)[0][0]
        max_q = np.max(self.q_values)
        td_target = reward + (self.gamma * max_q)
        td_error = td_target - self.q_values[action]
        self.q_values[action] += self.alpha * td_error


class EpsilonGreedyMultiQ:
    """
    ε-greedy Q-learner with three Q-tables, one for each value realization:
      state_id ∈ {-1, 0, +1}  ≈ {value - shock, value, value + shock}
    The bidder knows its current realization before bidding.
    """

    def __init__(self, name, value, a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
                 init_param=101.0, number_of_bids=19, bid_step=0.05, delta=0.0):
        self.name = name
        self.value = float(value)
        self.stochastic = float(delta)   # shock size ≥ 0
        self.a = a
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.time_step = 0

        self.number_of_bids = number_of_bids
        self.bid_options = np.array([i * bid_step for i in range(1, number_of_bids + 1)])

        optimism = float(init_param)
        self.q_values_map = {
            -1: np.full(self.number_of_bids, optimism),
             0: np.full(self.number_of_bids, optimism),
            +1: np.full(self.number_of_bids, optimism),
        }
        self.state_id = 0
        self.realized_value = self.value
        self.q_values = self.q_values_map[self.state_id]

    def begin_round(self):
        if self.stochastic > 0:
            r = random.random()
            if r < 1 / 3:
                self.state_id = -1
                self.realized_value = self.value - self.stochastic
            elif r < 2 / 3:
                self.state_id = 0
                self.realized_value = self.value
            else:
                self.state_id = +1
                self.realized_value = self.value + self.stochastic
            self.realized_value = max(0.0, self.realized_value)
        else:
            self.state_id = 0
            self.realized_value = self.value

        self.q_values = self.q_values_map[self.state_id]

    def place_bid(self):
        epsilon_t = self.a * np.exp(-self.b * self.time_step)
        self.time_step += 1

        qv = self.q_values_map[self.state_id]
        if np.random.rand() < epsilon_t:
            action_idx = np.random.randint(self.number_of_bids)
        else:
            max_v = np.max(qv)
            max_idx = np.flatnonzero(qv == max_v)
            action_idx = np.random.choice(max_idx)

        self._last_action_idx = action_idx
        self.q_values = qv
        return self.bid_options[action_idx]

    def update_strategy(self, bid, reward):
        a_idx = getattr(self, "_last_action_idx", None)
        if a_idx is None:
            a_idx = int(np.argmin(np.abs(self.bid_options - bid)))

        qv = self.q_values_map[self.state_id]
        max_q = np.max(qv)
        td_target = reward + self.gamma * max_q
        td_error = td_target - qv[a_idx]
        qv[a_idx] += self.alpha * td_error
        self.q_values = qv


class SarsaGreedy:
    """On-policy SARSA agent. Requires next_bid in update_strategy."""

    requires_next_action = True

    def __init__(self, name, value, a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
                 init_param=101, n_bids=19):
        self.name = name
        self.value = float(value)
        self.a, self.b = a, b
        self.alpha, self.gamma = alpha, gamma
        self.bid_options = np.array([i * 0.05 for i in range(1, n_bids + 1)])
        self.q_values = np.full(n_bids, float(init_param))
        self.n_bids = n_bids
        self.t = 0

    def _epsilon(self):
        return self.a * np.exp(-self.b * self.t)

    def _select_index(self):
        eps = self._epsilon()
        self.t += 1
        if np.random.rand() < eps:
            return np.random.randint(self.n_bids)
        m = self.q_values.max()
        ties = np.flatnonzero(self.q_values == m)
        return np.random.choice(ties)

    def place_bid(self):
        return self.bid_options[self._select_index()]

    def update_strategy(self, bid_t, reward_t, *, next_bid, terminal=False):
        a_t = np.where(self.bid_options == bid_t)[0][0]
        a_tp1 = np.where(self.bid_options == next_bid)[0][0]
        target = reward_t + self.gamma * self.q_values[a_tp1]
        td_err = target - self.q_values[a_t]
        self.q_values[a_t] += self.alpha * td_err


class ContextualLinUCB:
    """
    Per-arm contextual LinUCB over the fixed bid grid 0.05..0.95.
    Compatible with the framework: no next_bid; just place_bid() and update_strategy().
    """

    requires_next_action = False

    def __init__(self, name, value, d, alpha=1.0, reg=1.0, n_bids=19,
                 feature_fn=None, rng=None):
        self.name = name
        self.value = float(value)
        self.realized_value = getattr(self, "realized_value", self.value)

        self.d = int(d)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.n_bids = int(n_bids)
        self.bid_options = np.array([i * 0.05 for i in range(1, self.n_bids + 1)], dtype=float)

        if feature_fn is None:
            def feature_fn_default(bidder, bid):
                rv = getattr(bidder, "realized_value", bidder.value)
                return np.array([1.0, rv, bid, rv - bid], dtype=float)
            feature_fn = feature_fn_default
            self.d = 4

        self.feature_fn = feature_fn
        self.rng = rng if rng is not None else np.random.default_rng()

        self.A = np.array([self.reg * np.eye(self.d) for _ in range(self.n_bids)])
        self.b = np.zeros((self.n_bids, self.d), dtype=float)
        self._last_idx = None

    def begin_round(self):
        pass

    def _theta(self, a_idx):
        return np.linalg.solve(self.A[a_idx], self.b[a_idx])

    def _ucb(self, x, a_idx):
        Ainv_x = np.linalg.solve(self.A[a_idx], x)
        mean = x @ self._theta(a_idx)
        conf = self.alpha * np.sqrt(x @ Ainv_x)
        return mean + conf

    def place_bid(self):
        scores = np.empty(self.n_bids, dtype=float)
        for i, bid in enumerate(self.bid_options):
            x = self.feature_fn(self, float(bid))
            if x.shape[0] != self.d:
                raise ValueError(f"feature_fn produced dim {x.shape[0]} but LinUCB.d={self.d}")
            scores[i] = self._ucb(x, i)

        max_s = scores.max()
        winners = np.flatnonzero(scores == max_s)
        idx = int(self.rng.choice(winners))
        self._last_idx = idx
        return float(self.bid_options[idx])

    def update_strategy(self, bid, reward):
        try:
            a_idx = int(np.where(self.bid_options == bid)[0][0])
        except IndexError:
            diffs = np.abs(self.bid_options - float(bid))
            a_idx = int(np.argmin(diffs))
            if diffs[a_idx] > 1e-9:
                raise

        x = self.feature_fn(self, float(self.bid_options[a_idx])).reshape(self.d)
        self.A[a_idx] += np.outer(x, x)
        self.b[a_idx] += float(reward) * x
