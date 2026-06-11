# Algorithmic Collusion in First- and Second-Price Auctions

This repository accompanies a paper investigating whether reinforcement
learning algorithms tacitly collude in repeated auctions, and whether such
collusion is robust to asymmetries between bidders.

## Background

Reinforcement learning algorithms are increasingly used in high-frequency
trading, dynamic pricing, and online advertising auctions. Recent work has
shown these algorithms can learn to tacitly collude — a concerning outcome
given their growing role in markets. Banchio and Skrzypacz (2022) found that
symmetric Q-learning agents converge on collusive bids in first-price auctions
(FPAs), while converging on the static Nash equilibrium in second-price
auctions (SPAs).

This project tests whether those findings survive realistic perturbations:
asymmetric values, asymmetric learning-rate / discount / exploration
parameters, stochastic Markov values, expanded bid grids, and asymmetries in
the *algorithm* used (Q-learning vs. Sarsa vs. Contextual LinUCB).

## Main results

- **First-price auctions.** Tacit collusion between Q-learning agents is
  robust to every asymmetry tested — value gaps, parameter gaps, stochastic
  values, and pairings of different RL algorithms. All algorithm pairs
  (Q-learning × Sarsa × LinUCB) converged on low-bidding outcomes.

- **Second-price auctions, standard grid.** As expected, agents converge to
  the static Nash equilibrium across all asymmetries when bids are restricted
  to lie at most at the bidders' values.

- **Second-price auctions, expanded bid grid.** When the action space is
  extended above bidders' values, Q-learning agents can also sustain
  low-bidding collusive outcomes in SPAs — a setting previously thought to be
  collusion-proof. This is the main novel finding of the project.

The full write-up, with all heatmaps and discussion, is in
[`paper.pdf`](paper.pdf).

## Repository layout

```
.
├── paper.pdf            # thesis write-up
├── src/                 # auction framework
│   ├── agents.py        # QlearningGreedy, EpsilonGreedyMultiQ, SarsaGreedy, ContextualLinUCB
│   ├── environments.py  # FPA and SPA environments + convergence loop
│   └── simulation.py    # AuctionSimulation driver
├── simulations/         # experiment scripts (one per result section)
│   ├── single_simulation.py
│   ├── multiple_simulations_results.py
│   ├── parameter_grid.py
│   ├── noise_simulations.py
│   └── multiple_simulations_algo_asymmetry.py
├── notebooks/           # analysis and figure generation
├── data/                # CSV outputs from simulation runs
├── archive/             # earlier versions of the framework
└── req.txt              # Python dependencies
```

## Running

Install dependencies and run a script from the repo root:

```bash
pip install -r requirements.txt
python simulations/single_simulation.py
```

Notebooks under `notebooks/` reproduce the figures and analyses in
`paper.pdf`; launch Jupyter from the repo root so they can import from `src/`.
