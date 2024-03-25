"""
This is the discriminator network class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscriminatorNetwork(nn.Module):
    """
    This network is a convolutional neural network used for image classification. It will take two images as input and
    one as real and the other as generated. The network will output a single value which will be the probability of the
    image being real.
    """

    def __init__(self, image_size, depth_padding):
        """
        This is the constructor of the class.
        """
        super(DiscriminatorNetwork, self).__init__()

        self.conv1 = nn.Conv2d(2 + 1 + depth_padding * 2, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)

        self.fc1_size = 256 * (image_size // 16) * (image_size // 16)

        self.fc1 = nn.Linear(self.fc1_size, 1024)
        self.fc2 = nn.Linear(1024, 1)

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, brightfield, flourescent):
        """
        This method is used to forward pass the input through the network.
        :param x: torch.Tensor, input tensor
        :return: torch.Tensor, output tensor
        """
        x = torch.cat((brightfield, flourescent), 1)


        x = F.relu(self.conv1(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv3(x))
        x = self.max_pool(x)

        x = F.relu(self.conv4(x))
        x = self.max_pool(x)

        x = x.view(-1, self.fc1_size)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)

        return F.sigmoid(x)
