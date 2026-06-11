"""Run a single auction simulation and return its history."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import (
    QlearningGreedy,
    FPA_AuctionEnvironment,
    SPA_AuctionEnvironment,
    AuctionSimulation,
)

# Parameters
#   a          exploration constant
#   b          epsilon decay rate (beta)
#   alpha      learning rate
#   gamma      discount factor

bidder1 = QlearningGreedy(name="Agent1", value=1, a=0.025, b=0.0002,alpha=0.05, gamma=0.99, init_param=100)
bidder2 = QlearningGreedy(name="Agent2", value=1, a=0.025, b=0.0002,alpha=0.05, gamma=0.99, init_param=100)

simulation = AuctionSimulation(
    FPA_AuctionEnvironment,
    bidders=[bidder1, bidder2],
    max_rounds=1_000_000,
    convergence_limit=1000,
)

history = simulation.run()
