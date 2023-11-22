################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np

from matplotlib import pyplot as plt

from MyPCA import MyPCA

from hw4_utils import load_MNIST, convert_data_to_numpy, plot_points

np.random.seed(2023)

normalize_vals = (0.1307, 0.3081)

batch_size = 100

# load MNIST dataset
train_dataset, test_dataset, train_loader, test_loader = load_MNIST(batch_size, normalize_vals)

# convert to numpy
X, y = convert_data_to_numpy(train_dataset)

#####################
# ADD YOUR CODE BELOW
#####################

# specified number to reduce to
num_reduced_dims = 2

# initialize model
model = MyPCA(num_reduced_dims)

# fit model
model.fit(X)

# project data
x_proj = model.project(X)

# get list of colors for each label and plot accordingly
colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
for i, x in enumerate(x_proj):
    label = y[i]
    plt.plot(x[0], x[1], marker = "o", color = colors[label]) # i realized after how to do it correctly (in q2.py)

plt.savefig("hw4_q1.jpg")
