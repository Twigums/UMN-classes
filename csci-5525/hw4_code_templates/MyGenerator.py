import numpy as np

import torch
import torch.nn as nn

class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()

        # specified layers from randn input of size 128
        self.layers = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 784),
                nn.Tanh(),
                )

    def forward(self, x):
        return self.layers(x)
