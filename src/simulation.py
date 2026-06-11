"""Top-level simulation driver."""


class AuctionSimulation:
    """Controls a repeated auction simulation and stores results."""

    def __init__(self, environment_cls, bidders, max_rounds=10000, convergence_limit=1000):
        """
        :param environment_cls: A subclass of AbstractAuctionEnvironment (FPA or SPA).
        :param bidders: A list of bidder objects.
        """
        self.auction = environment_cls(bidders)
        self.max_rounds = max_rounds
        self.convergence_limit = convergence_limit

    def run(self):
        self.auction.run_auction(max_rounds=self.max_rounds,
                                 convergence_limit=self.convergence_limit)
        return self.auction.history
