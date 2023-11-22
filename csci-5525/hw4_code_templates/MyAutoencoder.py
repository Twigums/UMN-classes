import numpy as np

import torch
import torch.nn as nn

class MyAutoencoder(nn.Module):

    def __init__(self, max_epochs):
        super(MyAutoencoder, self).__init__()

        self.flatten = nn.Flatten() # flatten step

        # specified encoding step
        self.encoder = nn.Sequential(
                nn.Linear(784, 400),
                nn.Tanh(),
                nn.Linear(400, 2),
                nn.Tanh(),
                )

        # specified decoding step
        self.decoder = nn.Sequential(
                nn.Linear(2, 400),
                nn.Tanh(),
                nn.Linear(400, 784),
                nn.Sigmoid(),
                )

        self.max_epochs = max_epochs

    def fit(self, train_loader, criterion, optimizer):

        # for each epoch
        for i in range(self.max_epochs):
            epoch_loss = 0
            epoch_loss_list = []

            # j batch with (images, labels)
            for j, (images, labels) in enumerate(train_loader):

                # encode, then decode the encoded message
                flat_images = self.flatten(images)
                encoded = self.encoder(flat_images)
                decoded = self.decoder(encoded)

                # the loss will be the difference between the decoded message and the original input
                loss = criterion(decoded, flat_images)
                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

                # add the losses
                items = len(train_loader)
                epoch_loss += loss.item() / items

            # save losses for plotting
            epoch_loss_list.append(epoch_loss)

            print("EPOCH " + str(i + 1) + ", loss: " + str(epoch_loss) + ".")

            # convergence criteria: if improvement is <10% of original descent magnitude, stop
            if i == 1:
                slope_init = epoch_loss - last_loss

            elif i > 1:
                slope_current = epoch_loss - last_loss

                if abs(slope_init) * 0.1 > abs(slope_current):
                    print("Convergence criteria met, exiting fit.")
                    return epoch_loss_list

            last_loss = epoch_loss

        return epoch_loss_list

    # function for just encoding
    def encode(self, train_loader, num_images):

        encoded = []
        labels_list = []
        for j, (images, labels) in enumerate(train_loader):
            flat_images = self.flatten(images)
            encoded.append(self.encoder(flat_images))
            labels_list.append(labels)

        concat_encoded = torch.cat(encoded, dim = 0)
        labels_list = torch.cat(labels_list, dim = 0) # we need list of labels at correct indices for plotting purposes

        return concat_encoded, labels_list











