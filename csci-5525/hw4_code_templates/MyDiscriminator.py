import numpy as np

import torch
import torch.nn as nn

class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()

        # specified layers with MNIST input
        self.layers = nn.Sequential(
                nn.Flatten(), # redundantly flatten if necessary since real data is not flattened on input
                nn.Linear(784, 1024),
                nn.ReLU(),
                nn.Dropout(p = 0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p = 0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p = 0.3),
                nn.Linear(256, 1),
                nn.Sigmoid(),
                )

    # easy way to get losses without copy and pasting code in the q3.py file
    def get_losses(self, images, gt, criterion):
        output = self.forward(images)

        loss = criterion(output, gt)
        loss.backward()

        return loss

    def forward(self, x):
        return self.layers(x)
