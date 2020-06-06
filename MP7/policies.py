import numpy as np


class QPolicy:
    """
    Abstract policy to be subclassed
    """

    def __init__(self, statesize, actionsize, lr, gamma):
        self.statesize = statesize
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma

    def __call__(self, state, epsilon):
        qs = self.qvals(state[np.newaxis])[0]
        decision = np.random.uniform(0, 1)         # a random number between 0 and 1
        if decision < epsilon:
            pi = np.ones(self.actionsize) / self.actionsize  # even distributed probability over actions
        else:
            pi = np.zeros(self.actionsize)
            pi[np.argmax(qs)] = 1.0                # choose the action with highest Q value
        return pi

    def qvals(self, states):
        raise Exception("Not implemented")

    def td_step(self, state, action, reward, next_state, done):
        raise Exception("Not implemented")

    def save(self, outpath):
        raise Exception("Not implemented")

    def __str__(self):
        return self.__class__.__name__
