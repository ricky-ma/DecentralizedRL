import numpy as np


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
        self.SUBGRAPHS = np.random.permutation(subgraphs)
        self.NUM_AGENTS = self.SUBGRAPHS[0].size  # number of agents (size of adj matrix)
        self.NUM_ACTIONS = 3  # actions per agent
        self.TOTAL_STATES = 1000  # total states
        self.STATE = np.random.randint(self.TOTAL_STATES)
        self.DIM_FEATURES = 300  # value of d
        self.TOTAL_SAMPLES = 1e6
        self.DISCOUNT_GAMMA = 0.5

        '''
        sample generation
        '''
        # transition matrix
        self.mtx_prob_ssa = np.ones((self.TOTAL_STATES, self.TOTAL_STATES)) / self.TOTAL_STATES
        # reward generation
        self.reward_as = np.random.rand(self.TOTAL_STATES, self.NUM_AGENTS)

        '''
        feature generation
        '''
        feature = np.cos(np.pi * np.random.rand(self.DIM_FEATURES, self.TOTAL_STATES))
        feature_vec_norm = np.linalg.norm(feature, ord=2, axis=0)[np.newaxis, :]
        self.feature = feature / (np.ones((self.DIM_FEATURES, 1)) @ feature_vec_norm)

        '''
        weight + error traces
        '''
        self.WEIGHT = np.random.rand(self.DIM_FEATURES, self.NUM_AGENTS)
        self.ERROR = np.empty((1, ))  # vector to store error

    def update(self, subgraph, phi, phip, eta):
        """Update from new experience.
        Parameters
        ----------
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
        WEIGHT_next = np.zeros_like(self.WEIGHT)
        for agentn in range(self.NUM_AGENTS):
            for agentnn in range(self.NUM_AGENTS):
                b = np.asarray(subgraph.adjMatrix)[agentn][agentnn]
                r = self.reward_as[self.STATE, agentnn]
                wnn = self.WEIGHT[:, agentnn]
                WEIGHT_next[:, agentn] += b*(wnn + eta * phi * (r + np.dot(phip - phi, wnn)))

        error = (np.linalg.norm(self.WEIGHT, ord='fro') ** 2) / self.NUM_AGENTS
        self.ERROR = np.append(self.ERROR, values=[error])
        self.WEIGHT = WEIGHT_next
        return self.ERROR

    # def reset(self):
    #     """Reset weights, traces, and other parameters."""
    #     self.w = np.zeros(self.n)
    #     self.z = np.zeros(self.n)
