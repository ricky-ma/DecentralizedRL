import numpy as np
seed = 0


class DecentralizedTD:
    """Decentralized Temporal Difference Learning or TD(λ) with accumulating traces.
    The version implemented here uses general value functions (GVFs), meaning that
    the discount factor, γ, and the bootstrapping factor, λ, may be functions
    of state.
    If that doesn't seem germane to your problem, just use a constant value for them.
    """
    def __init__(self, subgraphs):
        """Initialize the learning algorithm.
        Parameters
        -----------
        subgraphs : List[Graph]
            List of decomposed graphs.
        """

        '''
        simulation setup
        '''
        np.random.seed(seed)
        self.SUBGRAPHS = np.random.permutation(subgraphs)
        self.NUM_AGENTS = self.SUBGRAPHS[0].shape[0]  # number of agents (size of adj matrix)
        self.TOTAL_STATES = 100  # total states
        np.random.seed(seed)
        self.STATE = np.random.randint(self.TOTAL_STATES)
        np.random.seed(seed)
        self.DIM_FEATURES = 300  # value of d
        self.EPOCHS = 20
        self.DISCOUNT_GAMMA = 0.9

        '''
        sample generation
        '''
        # transition matrix
        self.mtx_prob_ssa = np.ones((self.TOTAL_STATES, self.TOTAL_STATES)) / self.TOTAL_STATES
        # reward generation
        np.random.seed(seed)
        self.reward_as = np.random.rand(self.TOTAL_STATES, self.NUM_AGENTS)

        '''
        feature generation
        '''
        np.random.seed(seed)
        feature = np.cos(np.pi * np.random.rand(self.DIM_FEATURES, self.TOTAL_STATES))
        feature_vec_norm = np.linalg.norm(feature, ord=2, axis=0)[np.newaxis, :]
        self.feature = feature / (np.ones((self.DIM_FEATURES, 1)) @ feature_vec_norm)

        '''
        weight + error traces
        '''
        np.random.seed(seed)
        self.WEIGHT = np.random.rand(self.DIM_FEATURES, self.NUM_AGENTS)
        self.ERROR = np.zeros((self.EPOCHS * len(self.SUBGRAPHS) + 1, ))
        self.ERROR[0] = (np.linalg.norm(self.WEIGHT, ord='fro') ** 2) / self.NUM_AGENTS
        self.REWARD = np.zeros((self.EPOCHS * len(self.SUBGRAPHS) + 1,))

    def update(self, iteration, subgraph, phi, phip, eta):
        """Update from new experience.
        Parameters
        ----------
        iteration: int
            The current iteration (timestep).
        subgraph: Matrix[float]
            The n*n mixing matrix from the current timestep.
        phi : Vector[float]
            The feature vector for subgraph k.
        phip : Vector[float]
            The feature vector for subgraph k+1.
        eta : float
            The step-size parameter for updating the weight vector.
        Returns
        -------
        error : float
            The temporal difference error from the update.
        """
        acc_reward = 0
        WEIGHT_next = np.zeros_like(self.WEIGHT)
        for agentn in range(self.NUM_AGENTS):
            for agentnn in range(self.NUM_AGENTS):
                b = subgraph[agentn][agentnn]
                r = self.reward_as[self.STATE, agentnn]
                wnn = self.WEIGHT[:, agentnn]
                delta = b * (wnn + eta * phi * (r + np.dot(phip - phi, wnn)))
                WEIGHT_next[:, agentn] += delta
                acc_reward += r

        self.WEIGHT = WEIGHT_next
        self.ERROR[iteration] = (np.linalg.norm(self.WEIGHT, ord='fro') ** 2) / self.NUM_AGENTS
        self.REWARD[iteration] = self.REWARD[iteration-1] + acc_reward

    def reset(self):
        """Reset weights and errors."""
        np.random.seed(seed)
        self.WEIGHT = np.random.rand(self.DIM_FEATURES, self.NUM_AGENTS)
        self.ERROR = np.zeros((self.EPOCHS * len(self.SUBGRAPHS) + 1, ))
        self.REWARD = np.zeros((self.EPOCHS * len(self.SUBGRAPHS) + 1,))
