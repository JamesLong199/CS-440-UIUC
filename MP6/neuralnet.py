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
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
start_time = time.clock()
import matplotlib.pyplot as plt


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate,loss_fn,in_size,out_size):
        """
        Initialize the layers of your neural network

        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param yhat - an (N,out_size) tensor
            @param y - an (N,) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param in_size: Dimension of input
        @param out_size: Dimension of output

        For Part 1 the network should have the following architecture (in terms of hidden units):

        in_size -> 32 ->  out_size

        """

        super(NeuralNet, self).__init__()

        self.loss_fn = loss_fn
        self.lrate=lrate
        # self.linear1=nn.Linear(in_size,128)
        # self.linear2=nn.Linear(128,32)
        # self.linear3=nn.Linear(32,16)
        # self.linear4=nn.Linear(16,out_size)

        #self.net=nn.Sequential(nn.Conv2d(3,6,16),nn.ReLU(),nn.Conv2d(6,4,8),nn.ReLU(),nn.Conv2d(4,3,4),nn.ReLU(),nn.Linear(6*4*3,2))
        #self.net = nn.Sequential(nn.Linear(in_size,128),nn.ReLU(),nn.Linear(128,32,2),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,out_size))
        #self.net.train()
        #self.sigmoid=nn.Sigmoid()


        self.conv1 = nn.Conv2d(3, 15,3,1,1)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(7,6, 5)
        # self.conv3 = nn.Conv2d(6,5,5)
        self.fc1 = nn.Linear(15 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 25)
        self.fc3 = nn.Linear(25, out_size)

        # self.net = nn.Sequential(
        #     nn.Conv2d(3, 32, 5),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32),
        #
        #     nn.Conv2d(32, 64, 3),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(64),
        #
        #     nn.Conv2d(64, 64, 3),
        #     nn.MaxPool2d(2, 2),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(64),
        #
        #     nn.Linear(64, out_size),
        #     # nn.ReLU(True),
        #     # nn.BatchNorm2d(128)
        # )


    def set_parameters(self, params):
        """ Set the parameters of your network
        @param params: a list of tensors containing all parameters of the network
        """
        pass

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        #return self.parameters()
        pass


    def forward(self, x):
        """ A forward pass of your neural net (evaluates f(x)).

        @param x: an (N, in_size) torch tensor

        @return y: an (N, out_size) torch tensor of output from the network
        """
        # self.net.train()
        # return self.net(x)
        #return torch.ones(x.shape[0], 1)
        x=x.view(-1,3,32,32)
        #print(time.clock() - start_time, "seconds3")
        x = self.pool(F.relu(self.conv1(x)))
        #print("err1")
        #print("x1.shape is: ",x.shape)
        # x = self.pool(F.relu(self.conv2(x)))
        # #print("err2")
        # print("x2.shape is: ",x.shape)
        #
        # x = self.pool(F.relu(self.conv3(x)))
        # print("x3.shape is: ",x.shape)

        x = x.view(-1, 15 * 16 * 16)
        #print("err3")
        x = F.relu(self.fc1(x))
        #print("err4")
        x = F.relu(self.fc2(x))
        #print("err5")
        #x = F.relu(self.fc3(x))
        x = self.fc3(x)
        #print("done!")
        #print(x.shape)
        #print(time.clock() - start_time, "forward")
        return x

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        #x:true data y: true labels
        #self.net.train(False)
        #print(x.shape)
        criterion = nn.CrossEntropyLoss()
        #l_rate=0.01
        #optimizer = optim.SGD(self.parameters(), lr=self.lrate, momentum=0.9)
        optimizer = optim.Adam(self.parameters(), lr = self.lrate)
        # for i in range(len(x)):
            # labels=y[i]
            # inputs=x[i]
        optimizer.zero_grad()
        #y_predicted=self.net(x)

        #print(time.clock() - start_time, "seconds3")
        y_predicted=self.forward(x)
        #print(time.clock() - start_time, "seconds4")
        #inputs=x[]
        #outputs = NeuralNet(inputs)
        loss = criterion(y_predicted, y)
        loss.backward()
        optimizer.step()
        #print("loss is fine!")

        return loss.item()




def fit(train_set,train_labels,dev_set,n_iter,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, out_size) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M,) torch tensor
    @param n_iter: int, the number of epochs of training
    @param batch_size: The size of each batch to train on. (default 100)

    # return all of these:

    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object

    # NOTE: This must work for arbitrary M and N
    """

    in_size=32*32*3
    out_size=2
    net=NeuralNet(1e-5,nn.CrossEntropyLoss(),in_size,out_size)
    net.eval()
    dev_labels=[]
    losses=[]
    n_train=len(train_set)
    n_it=int(n_train/batch_size)
    mean_loss=0
    #data standarlization

    train_set=train_set.detach().numpy()

    for i in range(len(train_set)):
        temp=train_set[i]
        temp_std=np.std(temp)
        temp_mean=np.mean(temp)
        temp=(temp-temp_mean)/temp_std
    train_set=torch.from_numpy(train_set)
    n_iter=115
    print("n_iter is: ",n_iter)
    #train_set=train_set.view(-1,3,32,32)
    print(time.clock() - start_time, "seconds5")
    c = 0
    for n in range(n_iter):
        i=0
        while(i<train_set.shape[0]):
            c+=1
            i+=batch_size
    print("c is: ",c)
    for n in range(n_iter):
        print(time.clock() - start_time, "loop1")
        for i in range(n_it):
            #print(time.clock() - start_time, "loop2")
            it=i*batch_size
            if it+batch_size<len(train_set):
                the_images=train_set[it:it+batch_size]
                the_labels=train_labels[it:it+batch_size]
            else:
                the_images=train_set[it:len(train_set)-1]
                the_labels=train_labels[it:len(train_labels)-1]
            # y=forward(x_sets)
            #reshape
            mean_loss=net.step(the_images,the_labels)
            #print (time.clock() - start_time, "seconds0")
            #print("step is fine!")
        losses.append(mean_loss)
            #data=tuple((train_set[i],train_labels[i]))
# outputs=NeuralNet(y)
    print(time.clock() - start_time, "seconds6")

    #reshape tensor
    # train_set=train_set.detach().numpy()
    # for i in range(len(train_set)):
    #     train_set[i].reshape(32,32,3)
    # train_set=torch.from_numpy(train_set)


    #dev_set=dev_set.view(-1,3,32,32)
    print(time.clock() - start_time, "seconds0")
    outputs = net(dev_set)
    print (time.clock() - start_time, "seconds1")

    print("outputs is fine?")
    outputs=outputs.detach().numpy()
    print("outputs is: ",outputs)
    predicted=np.argmax(outputs,axis=1)
    print("predicted is: ",predicted)
    print("losses is: ",losses)
    x_axis=np.arange(len(losses))
    plt.plot(x_axis,losses)
    plt.show()
    #convert this to array
    #predicted=np.argmax
    #predicted=torch.argmax(outputs.data,1)
    #predicted=np.array(predicted)

    #print(predicted.type)
    #dev_labels.append(tuple((dev_set[i],predicted)))



    return losses,predicted,net
