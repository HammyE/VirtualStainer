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

import torch.distributed as dist
import torch.nn as nn

from train_model import train_model

# Defining the main function

data_dir = 'dataset'

def display_images():
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
        print("Memory usage of output: ", true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024,
              "MB")
        print("Memory usage of both: ",
              bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024 + true_flourescent.element_size() * true_flourescent.nelement() / 1024 / 1024,
              "MB")

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

if __name__ == '__main__':
    # Hyperparameters
    TILE_SIZE = 128
    DEPTH_PADDING = 2
    OVERLAP = TILE_SIZE // 2
    PIC_BATCH_SIZE = 3
    BATCH_SIZE = 8
    EPOCHS = 20
    LEARNING_RATE = 0.002
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_MODEL = True
    L1_LAMBDA = 0.1
    L2_LAMBDA = 0.01

    G_LR = 0.002
    D_LR = 0.01

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
    if '-lr' in sys.argv:
        LEARNING_RATE = float(sys.argv[sys.argv.index('-lr') + 1])

    if '-g_lr' in sys.argv:
        G_LR = float(sys.argv[sys.argv.index('-g_lr') + 1])

    if '-d_lr' in sys.argv:
        D_LR = float(sys.argv[sys.argv.index('-d_lr') + 1])

    if '-dn' in sys.argv:
        device_number = int(sys.argv[sys.argv.index('-dn') + 1])
        torch.cuda.set_device('cuda:' + str(device_number))
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
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    print("Data loader loaded.")

    if DEBUG:
        display_images()

    train_model(
        {'LEARNING_RATE': LEARNING_RATE,
         'TILE_SIZE': TILE_SIZE,
         'DEPTH_PADDING': DEPTH_PADDING,
         'MIN_ENCODER_DIM': MIN_ENCODER_DIM,
         'EPOCHS': EPOCHS,
         'loader': loader,
         'TRUE_BATCH_SIZE': TRUE_BATCH_SIZE,
         'PIC_BATCH_SIZE': PIC_BATCH_SIZE,
         'SAVE_MODEL': SAVE_MODEL,
         'L1_LAMBDA': L1_LAMBDA,
         'L2_LAMBDA': L2_LAMBDA,
         'DEVICE': DEVICE,
         'G_LR': G_LR,
         'D_LR': D_LR}
    )

    # extract materials

