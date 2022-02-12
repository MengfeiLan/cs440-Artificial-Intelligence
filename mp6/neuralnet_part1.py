# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP3. You should only modify code
within this file and neuralnet_part1.py, neuralnet_part2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        self.shadowneural = torch.nn.Sequential(torch.nn.Linear(in_size, 159), torch.nn.ReLU(), torch.nn.Linear(159, out_size))
    

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        result = self.shadowneural(x)
        return result

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        optimize = optim.SGD(self.parameters(), lr = self.lrate, momentum = 0.9)
        origin = y
        predict = self.forward(x)
        loss = self.loss_fn
        loss_new = loss(predict, origin)

        optimize.zero_grad()
        loss_new.backward()
        optimize.step()

        return loss_new.detach().cpu().numpy()

def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss_f = torch.nn.CrossEntropyLoss()
    lrate = 0.065
    in_size = train_set.size()[1]
    out_size = 4
    model = NeuralNet(lrate, loss_f, in_size, out_size)
    loss = []
    
    train_set_std = (train_set-train_set.mean())/(train_set.std())
    
    for epoch in range(epochs):
        print("epoch: ", epoch)
        for i in range(int(len(train_set_std)/batch_size) + 1):   
            batch = train_set_std[i * batch_size : (i + 1) * batch_size]
            label_batch = train_labels[i * batch_size : (i + 1) * batch_size]
            loss.append(model.step(batch, label_batch))


    #PATH = ''
    #torch.save(net.stat_dict(), PATH)
    print("train labels: ", train_labels)
    predictions = np.zeros(len(dev_set), dtype='int')
    dev_set_std = ( dev_set - dev_set.mean() ) / ( dev_set.std() )
    res = model(dev_set_std).detach().numpy()
    for i in range(len(res)):
        predictions[i] = np.int(np.argmax(res[i]))
        i += 1
    print("predictions: ", predictions)
    return loss, predictions, model
