import numpy as np
from scipy.special import softmax


class TD:
    """Temporal Difference Learning or TD(λ) with accumulating traces.
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
        self.SUBGRAPHS = subgraphs
        self.NUM_AGENTS = self.SUBGRAPHS[0].size  # number of agents (size of adj matrix)
        self.NUM_ACTIONS = 3  # actions per agent
        self.TOTAL_ACTIONS = int(np.power(self.NUM_ACTIONS, self.NUM_AGENTS))  # total actions
        self.TOTAL_STATES = 20  # total states
        self.DIM_PHI = self.TOTAL_ACTIONS * self.TOTAL_STATES  # |S||A|
        self.DIM_FEATURES = 30  # value of d
        self.VEC_EPSILON = np.random.rand(self.TOTAL_ACTIONS, 1)
        self.INIT_STATE = np.random.randint(self.TOTAL_STATES)
        self.TOTAL_SAMPLES = 1e6
        self.DISCOUNT = 0.5
        self.WEIGHT = np.zeros((self.DIM_FEATURES, self.NUM_AGENTS))

        '''
        sample generation
        '''
        # transition matrix
        self.mtx_prob_ssa = np.ones((self.TOTAL_STATES, self.TOTAL_STATES, self.TOTAL_ACTIONS)) / self.TOTAL_STATES
        # reward generation
        self.reward_as = np.random.rand(self.DIM_PHI, self.NUM_AGENTS)
        # behavior policy
        self.vec_prob_as = softmax(self.VEC_EPSILON)
        self.cdf_prob_as = np.cumsum(self.vec_prob_as)

        # self.z = np.mean(self.w, 2)

        '''
        feature generation
        '''
        self.feature = np.cos(np.pi * np.random.rand(self.DIM_FEATURES, self.DIM_PHI))
        feature_vec_norm = np.linalg.norm(self.feature, ord=2, axis=1)[:, np.newaxis]
        self.feature = self.feature / feature_vec_norm

    def get_value(self, s):
        """Get the approximate value for feature vector `phi`."""
        return np.dot(self.WEIGHT, s)

    def update(self, subgraph, idx_feature_curr, idx_feature_pred, eta):
        """Update from new experience.
        Parameters
        ----------
        subgraph: Vector[float]
            The n*n mixing matrix from the current timestep.
        idx_feature_curr : int
            Index of current feature vector.
        idx_feature_pred : Vector[int]
            Vector of indices of next feature vector for each agent.
        eta : float
            The step-size parameter for updating the weight vector.
        Returns
        -------
        delta : float
            The temporal difference error from the update.
        """

        delta = np.zeros(self.WEIGHT.shape)

        for agent in range(subgraph.size):
            r = self.reward_as[idx_feature_curr, agent]
            phi = self.feature[:, idx_feature_curr]
            phip = self.feature[:, idx_feature_pred[agent]]
            wk = self.WEIGHT[:, agent]
            temp = wk + eta * phi * (r + np.dot(phip - phi, wk))

            for col in range(subgraph.size):
                b = np.asarray(subgraph.adjMatrix)[agent][col]
                if b > 0:
                    delta[:, agent] += b * temp

        self.WEIGHT += delta
        print(self.WEIGHT)
        return delta

    # def reset(self):
    #     """Reset weights, traces, and other parameters."""
    #     self.w = np.zeros(self.n)
    #     self.z = np.zeros(self.n)
