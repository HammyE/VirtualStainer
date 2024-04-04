import subprocess

import matplotlib.pyplot as plt
import torch
from torch import multiprocessing
from torch.utils.data import DataLoader
from torchvision import transforms

from RangeTransform import RangeTransform
from dataset import HarmonyDataset, custom_collate_fn
from train_model import train_model

# Defining the main function

data_dir = 'dataset'



if __name__ == '__main__':

    try:
        # Set start method to 'spawn' for compatibility
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # start method can only be set once, ignore if set before

    # Hyperparameters
    TILE_SIZE = 128
    OVERLAP = TILE_SIZE // 2
    PIC_BATCH_SIZE = 4
    BATCH_SIZE = 8
    EPOCHS = 500
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_MODEL = True


    # For M1 Macs check for mps
    if DEVICE.type == 'cpu':
        if torch.backends.mps.is_available():
            DEVICE = torch.device("mps")
            # Caffeinate the process to prevent the Mac from sleeping
            subprocess.Popen('caffeinate')
        else:
            print("MPS device not found.")

    x = torch.ones(1, device=DEVICE)
    print(x)

    TRUE_BATCH_SIZE = BATCH_SIZE * PIC_BATCH_SIZE

    DEBUG = False


    # load data
    transform = transforms.Compose([
        RangeTransform(in_range=(0, 255), out_range=(0, 1)),
    ])

    print("Loading dataset...")



    print("Dataset loaded.")

    print("Loading data loader...")

    print("Data loader loaded.")


    learning_rate = [0.001, 0.01, 0.1]
    depth_padding = [2]
    l1_lambda = [0.01, 0.1, 1]
    l2_lambda = [0.01, 0.1, 1]

    parameter_sets = []
    for dp in depth_padding:
        dataset = HarmonyDataset(
            root=data_dir,
            equalization="histogram",
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            transform=transform,
            depth_padding=dp,
            picture_batch_size=PIC_BATCH_SIZE
        )
        for lr in learning_rate:
            for l1 in l1_lambda:
                for l2 in l2_lambda:
                    parameter_sets.append({
                        'LEARNING_RATE': lr,
                        'DEPTH_PADDING': dp,
                        'L1_LAMBDA': l1,
                        'L2_LAMBDA': l2,
                        'loader': DataLoader(
                            dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=custom_collate_fn,
                            shuffle=True,
                            num_workers=4
                        ),
                        'EPOCHS': EPOCHS,
                        'TRUE_BATCH_SIZE': TRUE_BATCH_SIZE,
                        'PIC_BATCH_SIZE': PIC_BATCH_SIZE,
                        'SAVE_MODEL': SAVE_MODEL,
                        'DEVICE': DEVICE
                    })

    n_cuda = torch.cuda.device_count()
    n_parameter_sets = len(parameter_sets)

    for i, params in enumerate(parameter_sets):
        params['DEVICE'] = torch.device(f'cuda:{i % n_cuda}')

        # Setup multiprocessing
    with multiprocessing.Pool(n_cuda) as pool:
        pool.map(train_model, parameter_sets)



