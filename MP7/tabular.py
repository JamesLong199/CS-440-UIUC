import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy


class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Stores the Q-value for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        if model is None:
            self.model = np.zeros(buckets + (actionsize,))
        else:
            self.model = model
        self.lr_model = np.zeros(buckets + (actionsize,))

    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], self.env.observation_space.high[1]]
        lower_bounds = [self.env.observation_space.low[0], self.env.observation_space.low[1]]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        ret = []
        for state in states:
            Q_vals = self.model[self.discretize(state)[0]][self.discretize(state)[1]]
            ret.append(Q_vals)

        return ret



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

        Q_curr, Q_next = self.qvals([state, next_state])
        # Q = self.qvals([state, next_state])
        Q_t = Q_curr[action]
        C = 75

        if done and next_state[0] >= 0.5:
                Q_local = 1

        else:
            max_Q = max(Q_next)
            Q_local = reward + self.gamma * max_Q

        loss = (Q_t - Q_local) * (Q_t - Q_local)
        N = self.lr_model[self.discretize(state)[0]][self.discretize(state)[1]][action]
        weighted_lr = min(self.lr, C / (C + N))
        # weighted_lr = self.lr
        Q_t += weighted_lr * (Q_local - Q_t)
        self.model[self.discretize(state)[0]][self.discretize(state)[1]][action] = Q_t
        # print("Q_t = ", Q_t)
        self.lr_model[self.discretize(state)[0]][self.discretize(state)[1]][action] += 1

        return loss


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('MountainCar-v0')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(30, 20), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/tabular.npy')
