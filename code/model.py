import numpy as np


class TD:
    """Temporal Difference Learning or TD(λ) with accumulating traces.
    The version implemented here uses general value functions (GVFs), meaning that
    the discount factor, γ, and the bootstrapping factor, λ, may be functions
    of state.
    If that doesn't seem germane to your problem, just use a constant value for them.
    Attributes
    ----------
    n : int
        The number of features (and therefore the length of the weight vector).
    z : Vector[float]
        The eligibility trace vector.
    w : Vector[float]
        The weight vector.
    """
    def __init__(self, n):
        """Initialize the learning algorithm.
        Parameters
        -----------
        n : int
            The number of features, i.e. expected length of the feature vector.
        """
        self.n = n
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def get_value(self, s):
        """Get the approximate value for feature vector `phi`."""
        return np.dot(self.w, s)

    def update(self, b, phi, r, phip, eta):
        """Update from new experience.
        Parameters
        ----------
        b: Vector[float]
            The n*n mixing matrix from the current timestep.
        phi : Vector[float]
            The observation/features from the current timestep.
        r : float
            The reward from the transition.
        phip : Vector[float]
            The observation/features from the next timestep.
        eta : float
            The step-size parameter for updating the weight vector.
        Returns
        -------
        delta : float
            The temporal difference error from the update.
        """
        delta = b * (self.w + eta * phi * (r + np.dot(phip - phi, self.w)))
        self.w += delta
        return delta

    def reset(self):
        """Reset weights, traces, and other parameters."""
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)
