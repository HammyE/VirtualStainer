'''This is the main file of the project. It will be used to run the project.'''
import time
import sys

import torch
# Importing the necessary modules
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from DiscriminatorNetwork import DiscriminatorNetwork
from GeneratorNetwork import GeneratorNetwork
from RangeTransform import RangeTransform
from dataset import HarmonyDataset, custom_collate_fn

# Defining the main function

data_dir = 'dataset'

if __name__ == '__main__':
    # Hyperparameters
    TILE_SIZE = 128
    DEPTH_PADDING = 2
    OVERLAP = TILE_SIZE // 2
    PIC_BATCH_SIZE = 4
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Take in command line arguments for batch size (-bs) and picture batch size (-pbs)
    if '-bs' in sys.argv:
        BATCH_SIZE = int(sys.argv[sys.argv.index('-bs') + 1])
    if '-pbs' in sys.argv:
        PIC_BATCH_SIZE = int(sys.argv[sys.argv.index('-pbs') + 1])


    # For M1 Macs check for mps
    if DEVICE.type == 'cpu':
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            x = torch.ones(1, device=DEVICE)
            print(x)
        else:
            print("MPS device not found.")

    TRUE_BATCH_SIZE = BATCH_SIZE * PIC_BATCH_SIZE

    DEBUG = False



    print("Running main.py")

    # load data
    transform = transforms.Compose([
        RangeTransform(in_range=(0, 255), out_range=(0, 1)),
    ])

    print("Loading dataset...")

    dataset = HarmonyDataset(
        root=data_dir,
        equalization="histogram",
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        transform=transform,
        depth_padding=DEPTH_PADDING,
        picture_batch_size=PIC_BATCH_SIZE
    )

    print("Dataset loaded.")

    print("Loading data loader...")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    print("Data loader loaded.")

    if DEBUG:
        # show data
        n_samples = 2
        range_transform = RangeTransform(in_range=(0, 1), out_range=(0, 255))
        print("Showing data...")

        images_processed = []
        for i, (bf_channels, true_flourescent) in enumerate(loader):

            bf_channels = range_transform(bf_channels)
            true_flourescent = range_transform(true_flourescent)
            if i >= n_samples:
                break
            print(bf_channels.shape)
            print(true_flourescent.shape)
            # calculate the memory usage of a batch
            print("Memory usage of input: ", bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024, "MB")
            print("Memory usage of output: ", true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024, "MB")
            print("Memory usage of both: ", bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024 + true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024, "MB")


            bf_img = bf_channels[0][DEPTH_PADDING]
            dead_img = true_flourescent[0][0]
            live_img = true_flourescent[0][1]

            print(dead_img.shape)
            # from tensor to numpy, then to cv2
            bf_img = bf_img.numpy()
            dead_img = dead_img.numpy()
            live_img = live_img.numpy()

            print(f"Max value of bf_img: {bf_img.max()}")
            print(f"Min value of bf_img: {bf_img.min()}")
            print(f"Max value of dead_img: {dead_img.max()}")
            print(f"Max value of live_img: {live_img.max()}")
            print(f"Min value of dead_img: {dead_img.min()}")
            print(f"Min value of live_img: {live_img.min()}")


            images_processed.append((bf_img, dead_img, live_img))

        fig, axs = plt.subplots(n_samples, 3, figsize=(5, 10), dpi=300)
        for i, (bf_img, dead_img, live_img) in enumerate(images_processed):

            axs[i][0].imshow(bf_img, cmap='gray')
            axs[i][1].imshow(dead_img, cmap='Greens')
            axs[i][2].imshow(live_img, cmap='Oranges')

        axs[0][0].set_title('Brightfield')
        axs[0][1].set_title('Dead')
        axs[0][2].set_title('Live')

        for ax in axs.flat:
            ax.set(xticks=[], yticks=[])

        plt.tight_layout()

        # Show the plot
        plt.show()


    # train model

    generator = GeneratorNetwork(out_channels=2,
                                 image_size=TILE_SIZE,
                                 depth_padding=DEPTH_PADDING,
                                 min_encoding_dim=MIN_ENCODER_DIM).to(DEVICE)

    discriminator = DiscriminatorNetwork(image_size=TILE_SIZE,
                                         depth_padding=DEPTH_PADDING).to(DEVICE)

    # GAN loss is
    # loss = E[log(D(x,y))] + E[log(1 - D(G(x, G(x)))]
    # where D is the discriminator, G is the generator, x is the real image, and z is the BF input to the generator
    # The first term is the loss for the discriminator, and the second term is the loss for the generator
    # The generator tries to minimize the second term, while the discriminator tries to maximize the first term
    # The generator tries to make the discriminator think that the generated image is real, while the discriminator tries
    # to distinguish between real and generated images

    g_loss_fn = torch.nn.BCELoss()
    d_loss_fn = torch.nn.BCELoss()


    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")
        for i, (bf_channels, true_flourescent) in enumerate(loader):
            start_time = time.time()
            bf_channels = bf_channels.to(DEVICE)
            true_flourescent = true_flourescent.to(DEVICE)

            print(f"Batch {i}")
            if epoch == 0 and i == 0:
                print(bf_channels.shape)
                print(true_flourescent.shape)
                # calculate the memory usage of a batch
                print("Memory usage of input: ", bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024, "MB")
                print("Memory usage of output: ", true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024, "MB")
                print("Memory usage of both: ", bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024 + true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024, "MB")

            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            outputs = generator(bf_channels)

            disc_labels_true_labels = torch.ones((TRUE_BATCH_SIZE, 1)).to(DEVICE)

            disc_true_outputs = discriminator(bf_channels, true_flourescent)
            disc_fake_outputs = discriminator(bf_channels, outputs)

            d_loss = (d_loss_fn(disc_true_outputs, disc_labels_true_labels) + d_loss_fn(disc_fake_outputs, 1 - disc_labels_true_labels)) / 2

            discriminator.zero_grad()

            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            disc_fake_outputs = discriminator(bf_channels, outputs)
            g_loss = g_loss_fn(disc_fake_outputs, disc_labels_true_labels)
            generator.zero_grad()

            g_loss.backward()
            g_optimizer.step()

            print(f"Generator loss: {g_loss}")
            print(f"Discriminator loss: {d_loss}")

            # Display one pair of real and generated images
            if i%10 == 0:
                fig = plt.figure(figsize=(10, 20))
                plt.subplot(3, 1, 1)
                image = bf_channels[0][DEPTH_PADDING].detach().cpu().numpy()
                plt.imshow(image, cmap='gray')
                plt.title('Brightfield')
                plt.axis('off')

                plt.subplot(3, 2, 3)
                image = true_flourescent[0][0].detach().cpu().numpy()
                plt.imshow(image, cmap='Greens')
                plt.title('Real Dead')
                plt.axis('off')

                plt.subplot(3, 2, 4)
                image = outputs[0][0].detach().cpu().numpy()
                plt.imshow(image, cmap='Greens')
                plt.title('Generated Dead')
                plt.axis('off')

                plt.subplot(3, 2, 5)
                image = true_flourescent[0][1].detach().cpu().numpy()
                plt.imshow(image, cmap='Oranges')
                plt.title('Real Live')
                plt.axis('off')

                plt.subplot(3, 2, 6)
                image = outputs[0][1].detach().cpu().numpy()
                plt.imshow(image, cmap='Oranges')
                plt.title('Generated Live')
                plt.axis('off')

                plt.show()

            print(f"Time taken for batch {i}: {round(time.time() - start_time, 2)} seconds")

            # test model

            # save model

            # extract materials

            pass


    # test model

    # save model

    # extract materials


    pass