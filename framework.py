import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class FPA_AuctionEnvironment:
    """Simulates a repeated first-price auction."""

    def __init__(self, bidders):
        self.bidders = bidders  # List of bidder objects
        self.history = []

    def run_auction(self, max_rounds=10000, convergence_limit=1000):
        """Run multiple rounds of the auction."""
        convergence_count = {bidder: 0 for bidder in self.bidders}  # Tracks stability count
        last_best_action = {bidder: np.argmax(bidder.q_values) for bidder in self.bidders}  # Initial argmax tracking
        
        for _ in range(max_rounds):  #'_' as loop index is not needed 
            ### Query - What should happen in case of draw ### I have that the winner is randomly drawn

            bids = {bidder: bidder.place_bid() for bidder in self.bidders}

            # In case of a draw (at the moment winner is randomly selected from winners --> surely this is a problem because both going low is very good)
            max_bid = max(bids.values())
            potential_winners = [bidder for bidder, bid in bids.items() if bid == max_bid]
            winner = random.choice(potential_winners)  # Randomly select a winner among ties
            winning_bid = bids[winner]
            
            # Initialize rewards for all bidders
            rewards = {bidder: 0 for bidder in self.bidders}  # (default 0)
            rewards[winner] = winner.value - winning_bid  # The winner's reward is assigned


            # Update all bidders' strategies
            for bidder in self.bidders:
                bidder.update_strategy(bids[bidder], rewards[bidder])  # Update Q-values

                # Track convergence: check if argmax(q_values) has remained the same
                current_best_action = np.argmax(bidder.q_values)
                if current_best_action == last_best_action[bidder]:
                    convergence_count[bidder] += 1
                else:
                    convergence_count[bidder] = 0  # Reset count if action changes
                
                last_best_action[bidder] = current_best_action  # Update last best action

            self.history.append((bids, winner, winning_bid, rewards[winner])) # final result, list of tuples giving info for EACH round

            # Check if both(all) bidders have converged
            if all(count >= convergence_limit for count in convergence_count.values()):
                print(f"Convergence detected after {_+1} rounds.")
                for bidder in self.bidders:
                    print(f"Bidder ", {bidder.name}, " had converged to ", np.argmax(bidder.q_values)*0.05)
                break  # Exit the loop if both bidders have converged

class EpsilonGreedy:
    """Represents an agent using the ε-greedy reinforcement learning strategy."""
    
    def __init__(self, name, value, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99): 
        self.name = name
        self.value = value  # The private value for the item
        self.a = a # the constant in front of the term for probability of exploring in every round
        self.b = b # the decay rate beta 
        self.time_step = 0  #initialise the time step keeping a count of rounds to 0
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # Discount factor for future rewards 
        self.number_of_bids = 19

        ## At the moment the available bid depends on the value of the bidder, might want to change this ## 
        self.bid_options = np.array([i*0.05 for i in range(1, self.number_of_bids + 1)]) #Creates the grid from 0 to value, num_actions giving density of discrete actions available
        
        #Optimistic initialisation

        optimism = float(1.5)
        self.q_values = np.full(self.number_of_bids, optimism)
        
        
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
    
    def __init__(self, bidders, max_rounds=10000, convergence_limit=1000):
        self.auction = FPA_AuctionEnvironment(bidders)
        self.max_rounds = max_rounds
        self.convergence_limit = convergence_limit

    def run(self):
        """Runs the auction and returns the history."""
        self.auction.run_auction(max_rounds=self.max_rounds, convergence_limit=self.convergence_limit)
        return self.auction.history

### RUNNING THE MODEL ### 

#Parameters

"""
  a = the constant in front of the term for probability of exploring in every round (epsilon)
  b = the decay rate beta 
  alpha = learning rate
  gamma = discount factor for future rewards

"""

#Define the bidder objects
bidder1 = EpsilonGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99)
bidder2 = EpsilonGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99)

#Run the simulation
simulation = AuctionSimulation(bidders=[bidder1, bidder2], max_rounds=1000000, convergence_limit=1000) # No. rounds of 1,000,000 is okay time wise

#history is a list of tuples giving info for each round (bids, winner, winning_bid, reward)
history = simulation.run()
bid_options = np.array([(i / (19 + 1)) for i in range(1, 19 + 1)]) ##value 1 is creating the available bid options for the heatmap. 

### PLOT THE RESULTS INTO A HEATMAP ###
### WORK IN PROGRESS ### 

## current problem -> the auction is converging immediately after the first thousand rounds when it really shouldn't be 