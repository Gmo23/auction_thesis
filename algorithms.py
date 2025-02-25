import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns

class FPA_AuctionEnvironment:
    """Simulates a repeated first-price auction."""

    def __init__(self, bidders):
        self.bidders = bidders  # List of bidder objects
        self.history = []

    def run_auction(self, rounds=1000):
        """Run multiple rounds of the auction."""
        for _ in range(rounds):  #'_' as loop index is not needed 
            bids = {bidder: bidder.place_bid() for bidder in self.bidders}
            winner = max(bids, key=bids.get)
            winning_bid = bids[winner]

            # The winner plays their bid and gets a reward (e.g., value - bid)
            reward = winner.value - winning_bid  
            winner.update_strategy(winning_bid, reward)
        
            self.history.append((bids, winner, winning_bid, reward)) # final result, list of tuples giving info for EACH round

class EpsilonGreedy:
    """Represents an agent using the ε-greedy reinforcement learning strategy."""
    
    def __init__(self, name, value, a=0.1, b=0.01, num_actions=10, alpha = 0.1, gamma = 0.9): #??? move action grid into AuctionEnvironment
        self.name = name
        self.value = value  # The private value for the item
        self.a = a # the constant in front of the term for probability of exploring in every round
        self.b = b # the decay rate beta 
        self.time_step = 0  #initialise the time step keeping a count of rounds to 0
        self.num_actions = num_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # Discount factor for future rewards 
        self.bid_options = np.array([(i / (self.num_actions + 1)) * self.value for i in range(1, self.num_actions + 1)]) #Creates the grid from 0 to value, num_actions giving density of discrete actions available
        self.q_values = np.zeros(num_actions)  # Q-values initialised at zero for each available action
        self.action_counts = np.zeros(num_actions)  # Times each bid was selected


    def place_bid(self):
        """Chooses a bid using an ε-greedy strategy."""
    
        #Find new probability of exploration given it is decaying in time
        epsilon_t = self.a * np.exp(-self.b * self.time_step)

        if np.random.rand() < epsilon_t: #explore
            action = np.random.randint(self.num_actions)  # Selects a random action
        else:   #exploit
            action = np.argmax(self.q_values)  # selects action corresponding to current heighest action

        self.time_step += 1  #increment time-step

        return self.bid_options[action]

    def update_strategy(self, bid, reward):
        """Updates bid strategy using the reward received."""
        action = np.where(self.bid_options == bid)[0][0]  # finds the index of the specific bid from the grid of available actions   

        #Compute TD update using alpha (learning rate)
        max_q = np.max(self.q_values) #Best future value
        td_target = (reward + self.gamma * max_q) # expected value (reward + discounted future best/greedy rewards)
        td_error = td_target - self.q_values[action]  # Difference from Q-value

        # update Q-value with learning rate alpha
        self.q_values[action] = self.q_values[action] + self.alpha * td_error

        # self.action_counts[action] += 1
        # self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]

class AuctionSimulation:
    """Controls the auction simulation and stores results."""
    
    def __init__(self, bidders, rounds=1000):
        self.auction = FPA_AuctionEnvironment(bidders)
        self.rounds = rounds

    def run(self):
        """Runs the auction and returns the history."""
        self.auction.run_auction(self.rounds)
        return self.auction.history

### RUNNING THE MODEL ### 

#Define the bidder objects
bidder1 = EpsilonGreedy(name="Agent1", value=1, a=0.1, b=0.01, num_actions=19, alpha = 0.1, gamma = 0.9)
bidder2 = EpsilonGreedy(name="Agent2", value=1, a=0.1, b=0.01, num_actions=19, alpha = 0.1, gamma = 0.9)

#Run the simulation
simulation = AuctionSimulation(bidders=[bidder1, bidder2], rounds=1000) # No. rounds of 100000 is okay time wise

#history is a list of tuples giving info for each round (bids, winner, winning_bid, reward)
history = simulation.run()