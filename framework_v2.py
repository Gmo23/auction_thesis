import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from abc import ABC, abstractmethod
from collections import deque
import inspect

def _best_action_signature(bidder):
    """
    Summarize a bidder's current 'best action(s)' for convergence checks.

    Returns
    -------
    tuple of ints
        - If bidder has multiple Q vectors (e.g., EpsilonGreedyMultiQ.q_values_map for states -1,0,+1),
          returns a 3-tuple of argmax indices in the fixed order (-1, 0, +1).
        - If bidder exposes a 2D q_values (num_bids x num_states), returns a tuple over columns.
        - Otherwise returns a 1-tuple with the single argmax (classic single Q-vector).
    """
    atol=1e-12

    # Case MultiQ class with a dict of per-state vectors
    if hasattr(bidder, "q_values_map"):
        cols = [-1, 0, +1]
        return tuple(int(np.argmax(bidder.q_values_map[c])) for c in cols)

    
    # LinUCB
    if hasattr(bidder, "A") and hasattr(bidder, "b") and hasattr(bidder, "feature_fn") \
       and hasattr(bidder, "alpha") and hasattr(bidder, "bid_options"):
        scores = []
        for i, bid in enumerate(bidder.bid_options):
            x = bidder.feature_fn(bidder, float(bid))
            theta = np.linalg.solve(bidder.A[i], bidder.b[i])
            mean  = float(x @ theta)
            Ainv_x = np.linalg.solve(bidder.A[i], x)
            bonus  = float(bidder.alpha * np.sqrt(x @ Ainv_x))
            scores.append(mean + bonus)
        scores = np.asarray(scores, dtype=float)
        mx = scores.max()
        top = tuple(np.flatnonzero(np.isclose(scores, mx, atol=atol)))
        return ("linucb", top)

    # q_values array
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
    # Prefer explicit flag; fall back to signature sniffing
    return getattr(agent, "requires_next_action", False) or \
           ("next_bid" in inspect.signature(agent.update_strategy).parameters)



class AbstractAuctionEnvironment(ABC):
    """Abstract base class for a repeated auction environment."""

    def __init__(self, bidders):
        """
        :param bidders: A list of bidder objects (e.g., EpsilonGreedy instances).
        """
        self.bidders = bidders
        self.history = []

    @abstractmethod
    def _compute_rewards(self, bids):
        """
        Given a dictionary of {bidder: bid}, compute:
         - The winning bidder (or bidders in a tie)
         - The winning bid
         - A dictionary of rewards for each bidder
        Returns: (winner, winning_bid, rewards_dict)
        """
        pass

    def run_auction(self, max_rounds=1000000, convergence_limit=1000):
        """
        Run multiple rounds of the auction, stopping early if convergence is detected.
        Works for bidders with a single Q-vector or multiple Q-vectors (e.g., MultiQ).
        """
        # Initialize convergence tracking using best-action signatures
        convergence_count = {bidder: 0 for bidder in self.bidders}
        last_best_sig     = {bidder: _best_action_signature(bidder) for bidder in self.bidders}
        queued_next = {}  # bidder -> bid to use next round (for SARSA/on-policy updates)

        for round_index in range(max_rounds):
            # Reveal realized values to bidders that support it (e.g., EpsilonGreedyMultiQ)
            for bidder in self.bidders:
                if hasattr(bidder, "begin_round"):
                    bidder.begin_round()

            # Each bidder places a bid
            # Current bids: use queued next if present (SARSA), else select now
            bids = {}
            for bidder in self.bidders:
                if bidder in queued_next:
                    bids[bidder] = queued_next.pop(bidder)
                else:
                    bids[bidder] = bidder.place_bid()
            

            # Determine winner(s), winning bid, and rewards
            winner, winning_bid, rewards = self._compute_rewards(bids)

            # For Sarsa/on-policy agents only
            next_bid = {}
            for bidder in self.bidders:
                if _needs_next_action(bidder):
                    nb = bidder.place_bid()       # on-policy: uses current Q
                    next_bid[bidder] = nb
                    queued_next[bidder] = nb      # becomes current bid next round

            # Update Q-values/strategies and check convergence
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

            # Snapshot Qs for analysis (handles 1D or 2D arrays; or keeps as-is if not ndarray)
            q_snapshot = {}
            for bidder in self.bidders:
                qv = getattr(bidder, "q_values", None)
                if isinstance(qv, np.ndarray):
                    q_snapshot[bidder.name] = qv.copy()
                else:
                    q_snapshot[bidder.name] = qv

            # Store full rewards dict (more useful downstream than only winner's reward)
            self.history.append((bids, winner.name, winning_bid, rewards.copy(), q_snapshot))

            if all(c >= convergence_limit for c in convergence_count.values()):
                print(f"Convergence detected after {round_index+1} rounds.")
                for bidder in self.bidders:
                    sig = last_best_sig[bidder]
                    kind, bid = sig

                    # If LinUCB (or any agent with bid_options + index set)
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
                        print(f"Bidder {bidder.name} converged to {(bid+1)*0.05}")
                break

            # Max-rounds message
            if round_index + 1 == max_rounds:
                print("Agents have not converged after 1,000,000 rounds")

class FPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Concrete environment for a repeated First-Price Auction."""

    def _compute_rewards(self, bids):
        max_bid = max(bids.values())
        potential_winners = [b for b, amt in bids.items() if amt == max_bid]
        winner = random.choice(potential_winners)
        winning_bid = bids[winner]

        rewards = {bidder: 0.0 for bidder in bids}

        # Use realized_value revealed in begin_round() if available
        w_value = getattr(winner, "realized_value", getattr(winner, "value", 0.0))
        rewards[winner] = float(w_value) - float(winning_bid)

        return winner, winning_bid, rewards

class SPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Concrete environment for a repeated Second-Price Auction."""

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

class AuctionSimulation:

    """Controls the auction simulation and stores results."""
    
    def __init__(self, environment_cls, bidders, max_rounds=10000, convergence_limit=1000):
        """
        :param environment_cls: A subclass of AbstractAuctionEnvironment (FPA or SPA).
        :param bidders: A list of Bidder objects.
        """
        self.auction = environment_cls(bidders)
        self.max_rounds = max_rounds
        self.convergence_limit = convergence_limit

    def run(self):
        """Runs the auction and returns the history."""
        self.auction.run_auction(max_rounds=self.max_rounds, convergence_limit=self.convergence_limit)
        return self.auction.history

class QlearningGreedy:
    """Represents an agent using the ε-greedy reinforcement learning strategy."""
    
    def __init__(self, name, value, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101, delta=0, num_bids=19): 
        self.name = name
        self.value = value
        self.a = a # the constant in front of the term for probability of exploring in every round
        self.b = b # the decay rate beta 
        self.time_step = 0  #initialise the time step keeping a count of rounds to 0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # Discount factor for future rewards 
        self.number_of_bids = num_bids
        self.init_param = init_param

        ### Value to handle the case with stochastic values 
        self.delta = delta

        ### Bad hard coding ### 
        self.bid_options = np.array([i*0.05 for i in range(1, self.number_of_bids + 1)]) # Creates the grid from 0 to 0.95
        
        #Optimisitic initialisations
        optimism = float(self.init_param)
        self.q_values = np.full(self.number_of_bids, optimism)
           


    def place_bid(self):
        """Chooses a bid using an ε-greedy strategy."""
    
        #Find new probability of exploration given it is decaying in time
        epsilon_t = self.a * np.exp(-self.b * self.time_step)
        
        if np.random.rand() < epsilon_t: #explore
            action = np.random.randint(self.number_of_bids) # Selects a random action

        else:   # exploit
            max_value = np.max(self.q_values)
            max_indices = np.flatnonzero(self.q_values == max_value)
            action = np.random.choice(max_indices) #in case of many q_values with max it randomly selects, e.g. at the start.

            # action = np.argmax(self.q_values)  # selects action corresponding to current heighest action
            
        self.time_step += 1  #increment time-step for decaying epsilon

        return self.bid_options[action]


    def update_strategy(self, bid, reward):
        """Updates bid strategy using the reward received."""
        action = np.where(self.bid_options == bid)[0][0]  # finds the index of the specific bid from the grid of available actions   

        #Compute TD update using alpha (learning rate)

        # Q(t+1) = [1-alpha]Q(t) + alpha[reward + gamma*maxQ(t)]
        # Q(t+1) = Q(t) + alpha[reward + gamma*maxQ(t) - Q(t)]
        # Q(t+1) = Q(t) + alpha[td_error]

        max_q = np.max(self.q_values) # = maxQ(t)
        td_target = reward + (self.gamma * max_q) # = [reward + gamma*maxQ(t)]
        td_error = td_target - self.q_values[action]  # = td_target - Q(t)

        # update Q-value with learning rate alpha
        self.q_values[action] = self.q_values[action] + self.alpha * td_error

class ContextualLinUCB:
    """
    Contextual bandit (LinUCB, per-arm) over the fixed bid grid 0.05..0.95.
    Compatible with your framework: no next_bid; just place_bid() and update_strategy().
    """
    requires_next_action = False  # like Q-learning/EpsilonGreedy it updates off-policy

    def __init__(self, name, value, d, alpha=1.0, reg=1.0, n_bids=19, feature_fn=None, rng=None):
        """
        name:  agent name (for logging)
        value: base/private value (env may update realized_value per round)
        d:     dimension of feature vectors x (must match feature_fn output length)
        alpha: exploration width (larger => more optimistic UCBs)
        reg:   ridge regularization for A_a initialization (A_a starts as reg * I)
        feature_fn: callable (bidder, bid_value) -> x (shape (d,))
        """
        self.name = name
        self.value = float(value)
        self.realized_value = getattr(self, "realized_value", self.value)  # env may overwrite per round

        self.d = int(d)
        self.alpha = float(alpha)
        self.reg = float(reg)
        self.n_bids = int(n_bids)
        self.bid_options = np.array([i * 0.05 for i in range(1, self.n_bids + 1)], dtype=float)

        if feature_fn is None:
            # Default: simple features using current realized_value and the candidate bid
            # x = [1, realized_value, bid, realized_value - bid]
            def feature_fn_default(bidder, bid):
                rv = getattr(bidder, "realized_value", bidder.value)
                return np.array([1.0, rv, bid, rv - bid], dtype=float)
            feature_fn = feature_fn_default
            self.d = 4  # override if default used

        self.feature_fn = feature_fn
        self.rng = rng if rng is not None else np.random.default_rng() #FOr tie breaking among equally good arms

        # Per-arm linear models: A[a] (dxd), b[a] (d)
        self.A = np.array([self.reg * np.eye(self.d) for _ in range(self.n_bids)])
        self.b = np.zeros((self.n_bids, self.d), dtype=float)

        # Optional: keep last chosen action index for diagnostics
        self._last_idx = None

    # Optional hook; your env already calls bidder.begin_round() if present.
    # You can use it if you want to precompute anything per round.
    def begin_round(self):
        # Example: no-op; environment may set self.realized_value before this
        pass

    def _theta(self, a_idx):
        # θ_a = A_a^{-1} b_a  (solve is more stable than explicit inverse)
        return np.linalg.solve(self.A[a_idx], self.b[a_idx])

    def _ucb(self, x, a_idx):
        Ainv_x = np.linalg.solve(self.A[a_idx], x)
        mean = x @ self._theta(a_idx)
        conf = self.alpha * np.sqrt(x @ Ainv_x)
        return mean + conf

    def place_bid(self):
        """
        Pick the bid with the highest UCB score for the *current* context.
        Context is read via feature_fn(self, bid_value), so the environment
        must have set any needed attributes (e.g., realized_value) beforehand.
        """
        scores = np.empty(self.n_bids, dtype=float)
        for i, bid in enumerate(self.bid_options):
            x = self.feature_fn(self, float(bid))
            if x.shape[0] != self.d:
                raise ValueError(f"feature_fn produced dim {x.shape[0]} but LinUCB.d={self.d}")
            scores[i] = self._ucb(x, i)

        # tie-break uniformly
        max_s = scores.max()
        winners = np.flatnonzero(scores == max_s)
        idx = int(self.rng.choice(winners))
        self._last_idx = idx
        return float(self.bid_options[idx])

    def update_strategy(self, bid, reward):
        """
        LinUCB update for the arm (bid) played this round:
        A_a <- A_a + x x^T
        b_a <- b_a + r x
        """
        # identify arm index
        try:
            a_idx = int(np.where(self.bid_options == bid)[0][0])
        except IndexError:
            # if 'bid' is a float with tiny rounding noise, match by tolerance
            diffs = np.abs(self.bid_options - float(bid))
            a_idx = int(np.argmin(diffs))
            if diffs[a_idx] > 1e-9:
                raise

        x = self.feature_fn(self, float(self.bid_options[a_idx]))
        x = x.reshape(self.d)

        # rank-1 updates
        self.A[a_idx] += np.outer(x, x)
        self.b[a_idx] += float(reward) * x

class EpsilonGreedyMultiQ:
    """
    ε-greedy Q-learner with *three* Q-tables, one for each value realization:
      state_id ∈ {-1, 0, +1}  ≈ {value - shock, value, value + shock}
    The bidder *knows* its current realization before bidding.
    """
    def __init__(self, name, value, a=0.025, b=0.0002, alpha=0.05, gamma=0.99,
                 init_param=101.0, number_of_bids=19, bid_step=0.05,
                 delta=0.0):
        self.name = name
        self.value = float(value)          # base value
        self.stochastic = float(delta)  # shock size ≥ 0
        self.a = a
        self.b = b
        self.alpha = alpha
        self.gamma = gamma
        self.time_step = 0

        self.number_of_bids = number_of_bids
        self.bid_options = np.array([i*bid_step for i in range(1, number_of_bids+1)])

        optimism = float(init_param)
        # one Q vector per state_id ∈ {-1,0,+1}
        self.q_values_map = {
            -1: np.full(self.number_of_bids, optimism),
             0: np.full(self.number_of_bids, optimism),
            +1: np.full(self.number_of_bids, optimism),
        }
        # Active state for this round (set in begin_round)
        self.state_id = 0
        self.realized_value = self.value

        # For compatibility with your snapshots/convergence checks
        self.q_values = self.q_values_map[self.state_id]

    # ---- round interface ----
    def begin_round(self):
        """
        Called by the environment *before* bids are placed.
        Draw the realized value:
          with prob 1/3: v - shock
          with prob 1/3: v
          with prob 1/3: v + shock
        """
        if self.stochastic > 0:
            r = random.random()
            if r < 1/3:
                self.state_id = -1
                self.realized_value = self.value - self.stochastic
            elif r < 2/3:
                self.state_id = 0
                self.realized_value = self.value
            else:
                self.state_id = +1
                self.realized_value = self.value + self.stochastic
            # (optional) keep values within [0, ∞)
            self.realized_value = max(0.0, self.realized_value)
        else:
            self.state_id = 0
            self.realized_value = self.value

        # update exposed active vector
        self.q_values = self.q_values_map[self.state_id]

    def place_bid(self):
        """
        ε-greedy on the *active* Q-vector.
        """
        epsilon_t = self.a * np.exp(-self.b * self.time_step)
        self.time_step += 1

        qv = self.q_values_map[self.state_id]
        if np.random.rand() < epsilon_t:  # explore
            action_idx = np.random.randint(self.number_of_bids)
        else:  # exploit (break ties randomly)
            max_v = np.max(qv)
            max_idx = np.flatnonzero(qv == max_v)
            action_idx = np.random.choice(max_idx)

        # stash for update convenience
        self._last_action_idx = action_idx

        # keep compatibility view
        self.q_values = qv
        return self.bid_options[action_idx]

    def update_strategy(self, bid, reward):
        """
        Standard bandit-style Q-learning update on the *active* Q-vector.
        Q(a) ← Q(a) + α [ r + γ max_{a'} Q(a') − Q(a) ].
        """
        # map bid to index (safe: bids are chosen from grid)
        a_idx = getattr(self, "_last_action_idx", None)
        if a_idx is None:
            # fallback mapping in case we didn't store it
            a_idx = int(np.argmin(np.abs(self.bid_options - bid)))

        qv = self.q_values_map[self.state_id]
        max_q = np.max(qv)
        td_target = reward + self.gamma * max_q
        td_error = td_target - qv[a_idx]
        qv[a_idx] += self.alpha * td_error

        # keep exposed active vector in sync
        self.q_values = qv

class SarsaGreedy:
    requires_next_action = True

    def __init__(self, name, value, a=0.025, b=0.0002, alpha=0.05, gamma=0.99, init_param=101, n_bids=19):

        self.name = name
        self.value = float(value)
        self.a, self.b = a, b
        self.alpha, self.gamma = alpha, gamma
        self.bid_options = np.array([i*0.05 for i in range(1, n_bids+1)])
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
        # continuing task: ignore 'terminal'
        a_t   = np.where(self.bid_options == bid_t)[0][0]
        a_tp1 = np.where(self.bid_options == next_bid)[0][0]
        target = reward_t + self.gamma * self.q_values[a_tp1]
        td_err = target - self.q_values[a_t]
        self.q_values[a_t] += self.alpha * td_err
