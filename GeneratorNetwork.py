import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from MaximumIntensityProjection import MaximumIntensityProjection


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

        return torch.sigmoid(x)


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

def generate_full_test(dataset, TILE_SIZE, OVERLAP, DEVICE, generator, display=False, well=None, return_images=False, debug=False):
    active_tiles, x_full, n_tiles, real_fluorescent, mask = dataset.get_well_sample(well)
    print("using well", well)
    print("Time to infer")
    # create 1080 x 1080 black image
    flourescent_image = np.zeros((3, 1080, 1080))
    bf_image = np.ones((1080, 1080))
    bf_anti_image = np.ones((1080, 1080))
    anti_image_done = False
    flourescent_list = []
    bf_list = []
    adjustment_matrix = np.zeros((1, 1080, 1080))
    filter_matrix = np.ones((1, TILE_SIZE, TILE_SIZE))
    adjustment_done = False
    tile_filters = {}
    adjustment_idx = 0
    for i in range(0, 1080, TILE_SIZE - OVERLAP):
        for j in range(0, 1080, TILE_SIZE - OVERLAP):
            tile = flourescent_image[:, i:i + TILE_SIZE, j:j + TILE_SIZE]
            okay = tile.shape[1] == TILE_SIZE and tile.shape[2] == TILE_SIZE

            if okay:
                if adjustment_idx in active_tiles:
                    adjustment_matrix[0, i:i + TILE_SIZE, j:j + TILE_SIZE] += filter_matrix[0]
                adjustment_idx += 1
    adjustment_idx = 0
    for i in range(0, 1080, TILE_SIZE - OVERLAP):
        for j in range(0, 1080, TILE_SIZE - OVERLAP):
            tile = flourescent_image[:, i:i + TILE_SIZE, j:j + TILE_SIZE]
            okay = tile.shape[1] == TILE_SIZE and tile.shape[2] == TILE_SIZE

            if okay:
                if adjustment_idx in active_tiles:
                    left_overlap = adjustment_matrix[0, i, j]
                    right_overlap = adjustment_matrix[0, i, j + TILE_SIZE - 1]
                    top_overlap = adjustment_matrix[0, i, j]
                    bottom_overlap = adjustment_matrix[0, i + TILE_SIZE - 1, j]

                    base_filter = np.ones((1, TILE_SIZE, TILE_SIZE))

                    if left_overlap > 0:
                        base_filter[:, :, 0:OVERLAP] = base_filter[:, :, 0:OVERLAP] * np.arange(0, 1, 1 / OVERLAP)
                    if right_overlap > 0:
                        base_filter[:, :, -OVERLAP:] = base_filter[:, :, -OVERLAP:] * np.arange(1, 0, -1 / OVERLAP)
                    if top_overlap > 0:
                        base_filter[:, 0:OVERLAP, :] = base_filter[:, 0:OVERLAP, :] * np.arange(0, 1, 1 / OVERLAP)[:,
                                                                                      np.newaxis]
                    if bottom_overlap > 0:
                        base_filter[:, -OVERLAP:, :] = base_filter[:, -OVERLAP:, :] * np.arange(1, 0, -1 / OVERLAP)[:,
                                                                                      np.newaxis]

                    tile_filters[adjustment_idx] = base_filter

                adjustment_idx += 1
    for level in range(20):
        flourescent_image = np.zeros((3, 1080, 1080))
        bf_image = np.ones((1080, 1080))
        tiles_used = 0
        tile_idx = 0
        x = x_full[n_tiles * level:n_tiles * (level + 1)].to(DEVICE)
        with torch.no_grad():
            if level == 10 and debug:
                # Save seven bf tiles for debugging
                for debug_image in range(7):
                    tile = x[debug_image]
                    plt.imshow(tile[2].cpu().numpy(), cmap='gray')
                    plt.axis('off')
                    plt.savefig(f"bf_tile_{debug_image}.png")
            y = generator(x)

            if level == 10 and debug:
                for debug_image in range(7):
                    tile = y[debug_image]
                    plt.imshow(tile[1].cpu().numpy(), cmap='Oranges')
                    plt.axis('off')
                    plt.savefig(f"live_tile_{debug_image}.png")
                    plt.imshow(tile[0].cpu().numpy(), cmap='Greens')
                    plt.axis('off')
                    plt.savefig(f"dead_tile_{debug_image}.png")

        for i in range(0, 1080, TILE_SIZE - OVERLAP):
            for j in range(0, 1080, TILE_SIZE - OVERLAP):
                tile = flourescent_image[:, i:i + TILE_SIZE, j:j + TILE_SIZE]

                okay = tile.shape[1] == TILE_SIZE and tile.shape[2] == TILE_SIZE

                if tile_idx in active_tiles and okay:
                    tile = y[tiles_used].detach().cpu().numpy()
                    bf_tile = x[tiles_used].detach().cpu().numpy()
                    flourescent_image[0:2, i:i + TILE_SIZE, j:j + TILE_SIZE] += tile * tile_filters[tile_idx]
                    bf_image[i:i + TILE_SIZE, j:j + TILE_SIZE] = bf_tile[2]
                    tiles_used += 1
                    if not anti_image_done:
                        bf_anti_image[i:i + TILE_SIZE, j:j + TILE_SIZE] = 0

                if okay:
                    tile_idx += 1

        anti_image_done = True

        # flourescent_image = flourescent_image / adjustment_matrix

        if level % 41 == 25:
            print(f"Dead range: {flourescent_image[0].min()} - {flourescent_image[0].max()}")
            print(f"Live range: {flourescent_image[1].min()} - {flourescent_image[1].max()}")

            plt.title(f"Level {level}")

            plt.subplot(1, 2, 1)

            plt.imshow(flourescent_image.T, vmax=1, vmin=0)
            plt.axis('off')

            plt.subplot(1, 2, 2)

            # Show the bf image
            plt.imshow(bf_image.T, cmap='gray', vmax=1, vmin=0)
            plt.axis('off')

            plt.show()

        flourescent_list.append(np.max(np.array([flourescent_image, np.zeros_like(flourescent_image)]), axis=0))
        bf_list.append(bf_image)
    # Readjust light intensity
    dead_min = 0
    dead_max = 3
    live_min = 0
    live_max = 7
    bf_min = 0
    bf_max = 50
    bf_list_scaled, live_list, dead_list = [], [], []
    for i in range(len(flourescent_list)):
        bf_list_scaled.append(bf_list[i] * 255.0)
        dead_list.append((flourescent_list[i][0] * 255.0 / (255.0 / (dead_max - dead_min))) + dead_min)
        live_list.append((flourescent_list[i][1] * 255.0 / (255.0 / (live_max - live_min))) + live_min)
        bf_list_scaled[i] = (bf_list_scaled[i] / (255.0 / (bf_max - bf_min))) + bf_min
    mip_trans = MaximumIntensityProjection(equalization_method="linear")
    bf, dead, live = mip_trans((bf_list_scaled, dead_list, live_list), mask=mask)
    bf_anti_image = bf_anti_image * np.quantile(bf, 0.99)
    bf = bf + bf_anti_image

    if debug:
        for depth in range(5, 12):
            plt.imshow(bf_list[depth], cmap='gray')
            plt.axis('off')
            plt.savefig(f"bf_full_{depth}.png")

            plt.imshow(live_list[depth], cmap='Oranges')
            plt.axis('off')
            plt.savefig(f"live_full_{depth}.png")

            plt.imshow(dead_list[depth], cmap='Greens')
            plt.axis('off')
            plt.savefig(f"dead_full_{depth}.png")



    bf_real, dead_real, live_real = mip_trans(real_fluorescent, mask=mask)

    if return_images:
        return bf, dead, live, bf_real, dead_real, live_real

    # live = 255 - live
    # dead = 255 - dead
    plt.figure(figsize=(15, 10), dpi=80)
    plt.subplot(2, 3, 1)
    plt.imshow(bf, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(2, 3, 2)
    plt.imshow(dead, cmap='Greens', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(2, 3, 3)
    plt.imshow(live, cmap='Oranges', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(2, 3, 4)
    plt.imshow(bf_real, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(2, 3, 5)
    plt.imshow(dead_real, cmap='Greens', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')
    plt.subplot(2, 3, 6)
    plt.imshow(live_real, cmap='Oranges', vmin=0, vmax=255)
    plt.colorbar()
    plt.axis('off')

    if display:
        plt.show()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
