import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class MyMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset,
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)
        super(MyMLP, self).__init__()
        self.layers = nn.Sequential(
                nn.Flatten(), # size needs to be 784 to multiply
                nn.Linear(
                    input_size,
                    hidden_size,
                    ), # input 28x28 -> 128 nodes
                nn.ReLU(),
                nn.Linear(
                    hidden_size,
                    output_size,
                    ), # input 128 -> 10 nodes
                )

        self.eta = learning_rate
        self.max_epochs = max_epochs

    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output
        return self.layers(x)


    def fit(self, train_loader, criterion, optimizer):
        '''
        Function used to train the MLP

        train_loader: includes the feature matrix and class labels corresponding to the training set,
        criterion: the loss function used,
        optimizer: which optimization method to train the model.
        '''

        # for convergence criteria
        last_loss = float("inf")
        delta_loss = float("inf")

        # Epoch loop
        for i in range(self.max_epochs):
            epoch_loss = 0
            total_data = 0
            correct_data = 0

            # Mini batch loop
            for j,(images,labels) in enumerate(train_loader):

                # Forward pass (consider the recommmended functions in homework writeup)
                output = self.forward(images)

                # Backward pass and optimize (consider the recommmended functions in homework writeup)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
                optimizer.zero_grad()

                # Track the loss and error rate
                items = len(train_loader)
                epoch_loss += loss.item() / items

                _, pred = torch.max(output.data, 1)
                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()

            # Print/return training loss and error rate in each epoch
            error_rate = 1 - (correct_data / total_data)

            print("EPOCH " + str(i + 1) + ", (loss, err_rate): (" + str(epoch_loss) + ", " + str(error_rate) + ").")
            # convergence criteria
            if i == 1: # second epoch, let's calculate initial slope over the first 2 epochs
                slope_init = epoch_loss - last_loss

            elif i > 1:
                slope_current = epoch_loss - last_loss

                if abs(slope_init) * 0.1 > abs(slope_current): # defined as 0.1 * initial slope (negligible improvement relative to initial step)
                    print("Convergence criteria met, exiting fit.")
                    return None

            last_loss = epoch_loss

    def predict(self, test_loader, criterion):
        '''
        Function used to predict with the MLP

        test_loader: includes the feature matrix and classlabels corresponding to the test set,
        criterion: the loss function used.
        '''

        with torch.no_grad(): # no backprop step so turn off gradients
            pred_loss = 0
            total_data = 0
            correct_data = 0

            for j,(images,labels) in enumerate(test_loader):

                # Compute prediction output and loss
                output = self.forward(images)
                loss = criterion(output, labels)

                # Measure loss and error rate and record
                items = len(test_loader)
                pred_loss += loss.item() / items

                _, pred = torch.max(output.data, 1)
                total_data += labels.size(0)
                correct_data += (pred == labels).sum().item()

        # Print/return test loss and error rate
        error_rate = 1 - (correct_data / total_data)
        print("PREDICTION (loss, err_rate): (" + str(pred_loss) + ", " + str(error_rate) + ").")

