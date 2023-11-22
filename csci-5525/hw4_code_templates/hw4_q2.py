################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize

from matplotlib import pyplot as plt

from MyAutoencoder import MyAutoencoder

from hw4_utils import load_MNIST, plot_points

np.random.seed(2023)

batch_size = 10

normalize_vals = (0.1307, 0.3081)

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

#####################
# ADD YOUR CODE BELOW
#####################

torch.manual_seed(2023)

max_epochs = 20
learning_rate = 0.001
num_images = 1000

# initialize model
model = MyAutoencoder(max_epochs)

# specify criterion and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
criterion = nn.MSELoss()

# fit the model
epoch_loss = model.fit(train_loader, criterion, optimizer)

# get projection and respective labels of images
encoded, labels = model.encode(train_loader, num_images)

# separate labels by color, get items, and then get random indices
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
items = encoded.size(0)
random_idx = torch.randperm(items)[:num_images] # choose random indices of num_images size

# select random data from random indices given above
random_x = encoded[random_idx].detach().numpy()
random_labels = labels[random_idx].detach().numpy()

# plot these chosen points on 2 dimensional projection
for i, x in enumerate(random_x):
    label = random_labels[i]
    x1 = random_x[i, 0]
    x2 = random_x[i, 1]

    plt.scatter(x1, x2, color = colors[label])

plt.savefig("hw4_q2_2.jpg")
