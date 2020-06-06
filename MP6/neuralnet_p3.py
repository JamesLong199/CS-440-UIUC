# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size,out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """
        super(NeuralNet, self).__init__()
        """
        1) DO NOT change the name of self.encoder & self.decoder
        2) Both of them need to be subclass of torch.nn.Module and callable, like
           output = self.encoder(input)
        3) Use 2d conv for extra credit part.
           self.encoder should be able to take tensor of shape [batch_size, 1, 28, 28] as input.
           self.decoder output tensor should have shape [batch_size, 1, 28, 28].
        """
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.in_size = in_size
        self.out_size = out_size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return xhat: an (N, out_size) torch tensor of output from the network.
                      Note that self.decoder output needs to be reshaped from
                      [N, 1, 28, 28] to [N, out_size] beforn return.
        """
        x = x.view(-1, 1, 28, 28)
        x_out = self.decoder(self.encoder(x))
        x_out = x_out.view(-1, self.out_size)
        return x_out

    def step(self, x):
        # x [100, 784]
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        criterion = self.loss_fn
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lrate)
        # x_in = x.view(-1, 1, 28, 28)
        x_out = self.forward(x)
        # x_out = x_out.view(-1, 784)
        loss = criterion(x_out, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

def fit(train_set,dev_set,n_iter,batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return xhats: an (M, out_size) NumPy array of reconstructed data.
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    train_length = len(train_set)

    lrate = 1e-3
    loss_fn = torch.nn.MSELoss()
    in_size = 784
    out_size = 784
    my_nn = NeuralNet(lrate, loss_fn, in_size, out_size)
    losses = []

    for i in range(n_iter):
        if (i + 1) * batch_size > train_length:
            break
        x = train_set[(i * batch_size % train_length): (i * batch_size % train_length) + batch_size]
        loss = my_nn.step(x)
        losses.append(loss)

    # PATH = "net_EC.model"
    # torch.save(my_nn.state_dict(), PATH)

    x_tensor = my_nn.forward(dev_set)
    xhats = x_tensor.detach().numpy()

    return losses,xhats, my_nn
