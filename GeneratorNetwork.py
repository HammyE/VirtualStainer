import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
    This class represents a double convolution layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        This is the constructor of the class.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        """
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        This method is used to forward pass the input through the network.
        :param x: torch.Tensor, input tensor
        :return: torch.Tensor, output tensor
        """
        return self.double_conv(x)


class DoubleConv2D(nn.Module):
    """
    This class represents a double convolution layer.
    """

    def __init__(self, in_channels, out_channels):
        """
        This is the constructor of the class.
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        """
        super(DoubleConv2D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        This method is used to forward pass the input through the network.
        :param x: torch.Tensor, input tensor
        :return: torch.Tensor, output tensor
        """
        return self.double_conv(x)


class GeneratorNetwork(nn.Module):
    """
    This class represents the generator network of the GAN model.
    """

    def __init__(
            self,
            out_channels=2,
            image_size=128,
            depth_padding=2,
            min_encoding_dim=16,
            features=None
    ):
        """
        This is the constructor of the class.
        """
        super(GeneratorNetwork, self).__init__()

        self.debug = False

        if features is None:
            features = [64, 128, 256]
        self.image_size = image_size
        self.depth_padding = depth_padding

        n_layers = 0
        im_dem = image_size
        while im_dem > min_encoding_dim:
            im_dem = im_dem // 2
            n_layers += 1


        if self.debug: print("Number of layers: ", n_layers)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool_2D = nn.MaxPool2d(2)

        in_channels = depth_padding * 2 + 1

        # Define the encoder
        for i in range(n_layers):
            if i == 0:
                self.downs.append(DoubleConv2D(in_channels, features[0]))
            elif i < len(features):
                self.downs.append(DoubleConv2D(features[i - 1], features[i]))
            else:
                features.append(features[-1])
                self.downs.append(DoubleConv2D(features[i - 1], features[i]))

        # Define the decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2D(feature * 2, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        if self.debug:
            print("Downs: ", self.downs)
            print("Ups: ", self.ups)
            print("Bottleneck: ", self.bottleneck)
            print("Final Conv: ", self.final_conv)

            print("Memory usage of the model: ", self.get_module_memory_usage() / 1024 / 1024, "MB")

    def get_module_memory_usage(self):

        mem_params = sum([param.nelement() for param in self.parameters()])
        mem_buffers = sum([buf.nelement() for buf in self.buffers()])
        return mem_params + mem_buffers

    def forward(self, x):
        """
        This method is used to forward pass the input through the network.
        :param x: torch.Tensor, input tensor
        :return: torch.Tensor, output tensor
        """
        if self.debug:
            print("Input shape: ", x.shape)
            print("Input memory usage: ", x.element_size() * x.nelement() / 1024 / 1024, "MB")

        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool_2D(x)

            if self.debug:
                print("Down shape: ", x.shape)
                print("Down memory usage: ", x.element_size() * x.nelement() / 1024 / 1024, "MB")

        x = self.bottleneck(x)

        if self.debug:
            print("Bottleneck shape: ", x.shape)
            print("Bottleneck memory usage: ", x.element_size() * x.nelement() / 1024 / 1024, "MB")

        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skip_connections[i // 2]
            x = torch.cat([x, skip], dim=1)
            x = self.ups[i + 1](x)

            if self.debug:
                print("Up shape: ", x.shape)
                print("Up memory usage: ", x.element_size() * x.nelement() / 1024 / 1024, "MB")

        x = self.final_conv(x)

        if self.debug:
            print("Output shape: ", x.shape)
            print("Output memory usage: ", x.element_size() * x.nelement() / 1024 / 1024, "MB")

        return x


# Initialize the UNet3D with 5 input channels and 2 output channels
if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    image_size = 64
    depth_padding = 2

    print("Creating the model...")
    test_generator = GeneratorNetwork(out_channels=2, image_size=image_size, depth_padding=depth_padding,
                                      min_encoding_dim=16)
    print("Model created.")

    # Create a dummy input with the correct shape
    print("Creating a dummy input...")
    dummy_input = torch.randn(32, depth_padding * 2 + 1, image_size, image_size)

    print("Input shape: ", dummy_input.shape)
    print("Input data type: ", dummy_input.dtype)
    print("Dummy input created.")

    print("Running the model...")
    res = test_generator(dummy_input)
    print("Model run successfully.")

    print("Output shape: ", res.shape)
    print("Output memory usage: ", res.element_size() * res.nelement() / 1024 / 1024, "MB")

    input("Press Enter to continue...")

    # Set up TensorBoard
    writer = SummaryWriter()

    # Add the model graph
    writer.add_graph(test_generator, dummy_input)
    writer.close()
