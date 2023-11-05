################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from MyCNN import MyCNN

from hw3_utils import load_MNIST

np.random.seed(2023)

batch_size = 32

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

# givens
input_size = 28 * 28
output_size = 10
kernel_size = 3
stride_size = 1
max_pool_size = 2

etas = [1e-5, 1e-4, 1e-3, 0.01, 0.1]
optim_models = [torch.optim.SGD, torch.optim.Adagrad, torch.optim.RMSprop, torch.optim.Adam]
optim_names = ["SGD"]

# number of epochs
max_epochs = 50

for i, optimizer_model in enumerate(optim_models):
    for learning_rate in etas:
        print("Using optimizer: " + optim_names[i] + ", eta: " + str(learning_rate))

        # initialize model
        model = MyCNN(input_size, output_size, kernel_size, stride_size, max_pool_size, learning_rate, max_epochs)

        # defines optimizer and criterion
        optimizer = optimizer_model(model.parameters(), lr = learning_rate)
        criterion = nn.CrossEntropyLoss()

        # fit the model
        model.fit(train_loader, criterion, optimizer)

        # predict
        model.predict(test_loader, criterion)
