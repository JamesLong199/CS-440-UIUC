import gym
import numpy as np
import torch
from torch import nn

import utils
from policies import QPolicy


def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    return nn.Sequential(nn.Linear(statesize, 64), nn.ReLU(), nn.Linear(64, actionsize))


class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        if model is None:
            self.model = make_dqn(statesize, actionsize)
        else:
            self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()      # indicates the model that nothing new is to be learnt and the model is used for testing
        with torch.no_grad():  # deactivate autograd engine
            states = torch.from_numpy(state).type(torch.FloatTensor)  # convert np.ndarray to torch.FloatTensor
            qvals = self.model(states)
        return qvals.numpy()   # convert tensor to numpy array

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        states = torch.from_numpy(state).type(torch.FloatTensor)
        next_states = torch.from_numpy(next_state).type(torch.FloatTensor)
        Q_curr = self.model(states)
        Q_t = Q_curr[action]

        Q_local = reward + self.gamma * torch.max(self.model(next_states)) * (1 - done)

        loss = self.criterion(Q_t, Q_local)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/dqn.model')
