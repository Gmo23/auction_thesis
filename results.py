from framework import (
    EpsilonGreedy,
    FPA_AuctionEnvironment,
    SPA_AuctionEnvironment,
    AuctionSimulation
)

### RUNNING THE MODEL ### 

#Parameters

"""
  a = the constant in front of the term for probability of exploring in every round (epsilon)
  b = the decay rate beta 
  alpha = learning rate
  gamma = discount factor for future rewards

"""

#Define the bidder objects
bidder1 = EpsilonGreedy(name="Agent1", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101)
bidder2 = EpsilonGreedy(name="Agent2", value=1, a=0.025, b=0.0002, alpha = 0.05, gamma = 0.99, init_param=101)

#run simulation for FPA or SPA
simulation = AuctionSimulation(FPA_AuctionEnvironment, bidders=[bidder1, bidder2], max_rounds=1000000, convergence_limit=1000) # No. rounds of 1,000,000 is okay time wise

#history is a list of tuples giving info for each round (bids, winner, winning_bid, reward)
history = simulation.run()



### PLOT THE RESULTS INTO A HEATMAP ###
### WORK IN PROGRESS ### 

## current problem -> the auction is converging immediately after the first thousand rounds when it really shouldn't be 

# Bertran games 
# more abstract intermediate classes 

"""To Do 
 - asymmetry in params
 - EXp 3 rl // Actor critic AC
 - Bertran // Asker paper
 - Pictures // visualisation he did of the delta in q-value and how you expect it to go positive and converge
 - CSV or Json (dictionary -> Json) saving data

 Papers:
 Base - https://arxiv.org/pdf/2202.05947
 RL - https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf
"""