'''This is the main file of the project. It will be used to run the project.'''
import os
import subprocess
import time
import sys

import numpy as np
import torch
import torchvision
# Importing the necessary modules
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
    PIC_BATCH_SIZE = 2
    BATCH_SIZE = 4
    EPOCHS = 300
    LEARNING_RATE = 0.002
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # For M1 Macs check for mps
    if DEVICE.type == 'cpu':
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            # Caffeinate the process to prevent the Mac from sleeping
            subprocess.Popen('caffeinate')
        else:
            print("MPS device not found.")

    # Take in command line arguments for batch size (-bs) and picture batch size (-pbs) and device (-d)
    if '-bs' in sys.argv:
        BATCH_SIZE = int(sys.argv[sys.argv.index('-bs') + 1])
    if '-pbs' in sys.argv:
        PIC_BATCH_SIZE = int(sys.argv[sys.argv.index('-pbs') + 1])
    if '-d' in sys.argv:
        u_device = sys.argv[sys.argv.index('-d') + 1]
        if u_device == 'cuda':
            DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif u_device == 'cpu':
            DEVICE = torch.device('cpu')

    x = torch.ones(1, device=DEVICE)
    print(x)




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
        for batch_idx, (bf_channels, true_flourescent) in enumerate(loader):

            bf_channels = range_transform(bf_channels)
            true_flourescent = range_transform(true_flourescent)
            if batch_idx >= n_samples:
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
        for batch_idx, (bf_img, dead_img, live_img) in enumerate(images_processed):

            axs[batch_idx][0].imshow(bf_img, cmap='gray')
            axs[batch_idx][1].imshow(dead_img, cmap='Greens')
            axs[batch_idx][2].imshow(live_img, cmap='Oranges')

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

    # Add writers for tensorboard
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    dead_writer = SummaryWriter(f"runs/{time_stamp}/dead")
    live_writer = SummaryWriter(f"runs/{time_stamp}/live")
    bf_writer = SummaryWriter(f"runs/{time_stamp}/brightfield")
    progress_writer = SummaryWriter(f"runs/{time_stamp}/progress")

    dataset.__getitem__(12123)



    for epoch in range(EPOCHS):
        logging_time = 0
        print(f"Epoch {epoch}")
        for batch_idx, (bf_channels, true_flourescent) in enumerate(loader):
            start_time = time.time()
            bf_channels = bf_channels.to(DEVICE)
            true_flourescent = true_flourescent.to(DEVICE)

            print(f"Batch {batch_idx}")
            if epoch == 0 and batch_idx == 0:
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


            # Display one pair of real and generated images
            if time.time() - logging_time > 60:
                print(f"Time taken for batch {batch_idx}: {round(time.time() - start_time, 2)} seconds\n",
                      f"Discriminator loss: {d_loss.item()}, Generator loss: {g_loss.item()}"
                    )

                logging_time = time.time()

                # extract indeces for 4 different images

                indeces = np.arange(0, 4) * PIC_BATCH_SIZE
                dead_sample = true_flourescent[indeces, 0]
                live_sample = true_flourescent[indeces, 1]
                bf_sample = bf_channels[indeces, DEPTH_PADDING]

                dead_sample_gen = outputs[indeces, 0]
                live_sample_gen = outputs[indeces, 1]

                # create channels to accomodate colors
                dead_sample_gen = dead_sample_gen.view(-1, 1, TILE_SIZE, TILE_SIZE)
                dead_sample_gen = torch.cat((dead_sample_gen*0.1, dead_sample_gen*0.8, dead_sample_gen*0), 1)

                live_sample_gen = live_sample_gen.view(-1, 1, TILE_SIZE, TILE_SIZE)
                live_sample_gen = torch.cat((live_sample_gen*0.9, live_sample_gen*0.8, live_sample_gen*0), 1)

                dead_sample = dead_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
                dead_sample = torch.cat((dead_sample*0.1, dead_sample*0.8, dead_sample*0), 1)

                live_sample = live_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
                live_sample = torch.cat((live_sample*0.9, live_sample*0.8, live_sample*0), 1)

                bf_sample = bf_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)


                print(dead_sample_gen.shape)

                dead_real_grid = torchvision.utils.make_grid(dead_sample)
                live_real_grid = torchvision.utils.make_grid(live_sample)
                bf_real_grid = torchvision.utils.make_grid(bf_sample)
                dead_fake_grid = torchvision.utils.make_grid(dead_sample_gen)
                live_fake_grid = torchvision.utils.make_grid(live_sample_gen)

                dead_writer.add_image('Real', dead_real_grid, epoch)
                live_writer.add_image('Real', live_real_grid, epoch)
                bf_writer.add_image('Real', bf_real_grid, epoch)
                dead_writer.add_image('Generated', dead_fake_grid, epoch)
                live_writer.add_image('Generated', live_fake_grid, epoch)

                # log losses
                progress_writer.add_scalar('Discriminator Loss', d_loss.item(), epoch)
                progress_writer.add_scalar('Generator Loss', g_loss.item(), epoch)

                # accuracy
                disc_true_outputs = disc_true_outputs.detach().cpu().numpy()
                disc_fake_outputs = disc_fake_outputs.detach().cpu().numpy()
                disc_true_outputs = np.round(disc_true_outputs)
                disc_fake_outputs = np.round(disc_fake_outputs)
                true_accuracy = np.sum(disc_true_outputs) / len(disc_true_outputs)
                fake_accuracy = np.sum(disc_fake_outputs) / len(disc_fake_outputs)
                progress_writer.add_scalar('True Accuracy', true_accuracy, epoch)
                progress_writer.add_scalar('Fake Accuracy', fake_accuracy, epoch)






                if DEBUG:
                    fig = plt.figure(figsize=(6, 12))
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

                    plt.savefig(f"runs/images_25_march/epoch_{epoch}_batch_{batch_idx}.png")



            # test model

            # save model

            # extract materials

            pass


    # test model

    # save model

    # save model
    torch.save(generator.state_dict(), f"runs/generator_{time_stamp}.pt")
    torch.save(discriminator.state_dict(), f"runs/discriminator_{time_stamp}.pt")


    # extract materials


    pass