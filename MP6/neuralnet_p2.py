# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss functions
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.net = torch.nn.Sequential(torch.nn.Linear(in_size, 256), torch.nn.ReLU(), torch.nn.Linear(256, out_size))

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2)
        # )
        #
        # self.linear_layer = nn.Sequential(
        #     nn.Linear(16 * 14 * 14, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, out_size)
        # )





    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.net.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        # x = x.view(-1, 1, 28, 28)
        # out = self.layer1(x)
        # out = out.view(-1, 16 * 14 * 14)
        # out = self.linear_layer(out)
        # return out
        return self.net(x)


    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        criterion = self.loss_fn
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrate, weight_decay=0.002)
        y_pred = self.forward(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    mean = train_set.mean(dim = -2, keepdim = True)
    std = train_set.std(dim = -2, keepdim = True)
    train_set = (train_set - mean) / std
    dev_set = (dev_set - mean) / std

    train_length = len(train_set)

    lrate = 5e-4
    loss_fn = torch.nn.CrossEntropyLoss()
    in_size = 784
    out_size = 5
    my_nn = NeuralNet(lrate, loss_fn, in_size, out_size)
    losses = []

    for i in range(n_iter):
        x = train_set[(i * batch_size % train_length):((i + 1) * batch_size % train_length)]
        y = train_labels[(i * batch_size % train_length):((i + 1) * batch_size % train_length)]
        loss = my_nn.step(x, y)
        losses.append(loss)

    y_tensor = my_nn.forward(dev_set)
    yhats = np.empty([len(dev_set)])
    for i in range(len(dev_set)):
        y_max = torch.argmax(y_tensor[i])
        yhats[i] = y_max

    return losses, yhats, my_nn
