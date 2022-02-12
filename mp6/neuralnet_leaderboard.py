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
        self.cnn = torch.nn.Sequential(

            torch.nn.Conv2d(3, 30, kernel_size=3), 
            torch.nn.BatchNorm2d(30),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(3, 3),
            torch.nn.Conv2d(30, 60, 3),
            torch.nn.BatchNorm2d(60),
            torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(60,90, 3),
            torch.nn.BatchNorm2d(90),
            torch.nn.LeakyReLU(),
            # torch.nn.Conv2d(135,180, 2),
            # torch.nn.BatchNorm2d(180),
            # torch.nn.LeakyReLU(),
            # torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(3240,128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128,out_size))




    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        x_transformed = x.view(x.shape[0],3,32,32)
        result = self.cnn(x_transformed)
        return result

    def step(self, x,y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        optimize = optim.SGD(self.parameters(), lr = self.lrate, momentum = 0.9)
        # optimize = torch.optim.Adam(self.parameters(), lr=0.001)
        origin = y
        predict = self.forward(x)
        loss = self.loss_fn
        l2_norm = 0
        for parameter in self.parameters():
            l2_norm += torch.norm(parameter) * torch.norm(parameter)
        lambda_value = 0.00001
        loss_new = loss(predict, origin)  + lambda_value * l2_norm
        optimize.zero_grad()
        loss_new.backward()
        optimize.step()
        result_loss = loss_new.detach().cpu().numpy() 

        return np.array(result_loss, dtype="float32")

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

    model.train()
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
    predictions = np.zeros(len(dev_set), dtype='int')
    dev_set_std = ( dev_set - dev_set.mean() ) / ( dev_set.std() )
    res = model(dev_set_std).detach().numpy()
    for i in range(len(res)):
        predictions[i] = np.int(np.argmax(res[i]))
        i += 1
    print("loss: ", loss)
    return loss, predictions, model
