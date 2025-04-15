import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from abc import ABC, abstractmethod

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

    def run_auction(self, max_rounds=10000, convergence_limit=1000):
        """
        Run multiple rounds of the auction, stopping early if convergence is detected.
        Child classes can override or extend this method as needed.
        """
        # Track how many consecutive rounds each bidder has remained on the same best action
        convergence_count = {bidder: 0 for bidder in self.bidders}
        # Track each bidder's previous best action
        last_best_action = {bidder: np.argmax(bidder.q_values) for bidder in self.bidders}

        for round_index in range(max_rounds):
            # Each bidder places a bid
            bids = {bidder: bidder.place_bid() for bidder in self.bidders}

            # Use the child-class method to determine winner(s), winning bid, and rewards
            winner, winning_bid, rewards = self._compute_rewards(bids)

            # Update Q-values/strategies
            for bidder in self.bidders:
                bidder.update_strategy(bids[bidder], rewards[bidder])

                # Convergence check
                current_best_action = np.argmax(bidder.q_values)
                if current_best_action == last_best_action[bidder]:
                    convergence_count[bidder] += 1
                else:
                    convergence_count[bidder] = 0

                last_best_action[bidder] = current_best_action

            # Record round results (list of tuples giving info for EACH round)
            self.history.append((bids, winner, winning_bid, rewards[winner]))

            # If all bidders have kept the same best action for `convergence_limit` rounds, stop
            if all(count >= convergence_limit for count in convergence_count.values()):
                print(f"Convergence detected after {round_index + 1} rounds.")
                for bidder in self.bidders:
                    print(f"Bidder", bidder.name, "had converged to ", np.argmax(bidder.q_values)*0.05)  #very ugly hard-coding CHANGE

                
                break

class FPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Concrete environment for a repeated First-Price Auction."""

    def _compute_rewards(self, bids):
        """
        - Winner is the highest bidder.
        - In ties, pick a random winner among those tied for highest bid.
        - Winner's reward = winner.value - winning_bid
        """
        max_bid = max(bids.values())
        potential_winners = [bidder for bidder, bid_amount in bids.items() if bid_amount == max_bid]

        winner = random.choice(potential_winners)  # break tie randomly
        winning_bid = bids[winner]

        # Initialize reward dictionary
        rewards = {bidder: 0 for bidder in bids}
        rewards[winner] = winner.value - winning_bid  # first-price payoff (winner.value is 1 by default it is left open for later asymmetry)

        return winner, winning_bid, rewards

class SPA_AuctionEnvironment(AbstractAuctionEnvironment):
    """Concrete environment for a repeated Second-Price Auction."""

    def _compute_rewards(self, bids):
        """
        - Winner is the highest bidder.
        - In ties, pick a random winner among the tied highest bidders and they pay that bid i.e. reduces to a FPA
        - Winner's payoff = winner.value - second_highest_bid
        """
        # Identify highest bid
        max_bid = max(bids.values())
        potential_winners = [bidder for bidder, bid_amount in bids.items() if bid_amount == max_bid]
        winner = random.choice(potential_winners)  # tie-breaking

        # Identify second-highest bid (or highest among losers)
        other_bids = [bid_amount for b, bid_amount in bids.items() if b != winner]
        second_price = max(other_bids) if other_bids else 0 

        # Assign rewards
        rewards = {bidder: 0 for bidder in bids}
        winning_bid = bids[winner]
        rewards[winner] = winner.value - second_price

        return winner, winning_bid, rewards

class EpsilonGreedy:
    """Represents an agent using the ε-greedy reinforcement learning strategy."""
    
    def __init__(self, name, value, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=1): 
        self.name = name
        self.value = value  # The private value for the item
        self.a = a # the constant in front of the term for probability of exploring in every round
        self.b = b # the decay rate beta 
        self.time_step = 0  #initialise the time step keeping a count of rounds to 0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # Discount factor for future rewards 
        self.number_of_bids = 19
        self.init_param = init_param

        ## At the moment the available bid depends on the value of the bidder, might want to change this ## 
        self.bid_options = np.array([i*0.05 for i in range(1, self.number_of_bids + 1)]) #Creates the grid from 0 to value, num_actions giving density of discrete actions available
        
        ###############################

        #Optimistic initialisation

        optimism = float(self.init_param)
        self.q_values = np.full(self.number_of_bids, optimism)
        
        ###############################
        
        # self.q_values = np.zeros(self.number_of_bids)  # Q-values initialised at zero for each available action
        # self.action_counts = np.zeros(num_actions)  # Times each bid was selected


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
        max_q = np.max(self.q_values) #Best future value
        td_target = reward + (self.gamma * max_q) # expected value (reward + discounted future best/greedy rewards)
        td_error = td_target - self.q_values[action]  # Difference from Q-value

        # update Q-value with learning rate alpha
        self.q_values[action] = self.q_values[action] + self.alpha * td_error
        

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
