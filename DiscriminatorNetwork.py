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

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)


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


        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.max_pool(x)


        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.max_pool(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.max_pool(x)

        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = self.max_pool(x)

        x = x.view(-1, self.fc1_size)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)

        return F.sigmoid(x)

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        # Define the convolutional layers
        self.model = nn.Sequential(
            # Input: (input_channels) x 128 x 128
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),  # (ndf) x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),  # (ndf*2) x 32 x 32
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),  # (ndf*4) x 16 x 16
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),  # (ndf*8) x 8 x 8
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),  # (1) x 7 x 7

            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    # Create a random input tensor
    input = torch.randn(1, 7, 128, 128)

    # Create an instance of the PatchGANDiscriminator
    discriminator = PatchGANDiscriminator(input_channels=7)

    # Perform a forward pass
    output = discriminator(input)

    # Print the output tensor
    #print(output)

    # Print the shape of the output tensor
    print(output.shape)
