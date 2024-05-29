import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import multiprocessing
from torchvision import transforms
from multiprocessing import Process, Manager

from RangeTransform import RangeTransform
from dataset import HarmonyDataset, custom_collate_fn
from train_model import train_model

# Defining the main function

data_dir = 'dataset'


def worker_func(shared_param_sets, lock, gpu_id):
    torch.cuda.set_device(gpu_id)  # Set GPU for this process
    while True:
        lock.acquire()
        if not shared_param_sets:
            lock.release()
            break  # Exit loop if there are no more parameter sets
        params = shared_param_sets.pop(0)  # Safely pop the next parameter set
        lock.release()

        # Now, train the model with these parameters on the assigned GPU
        params['DEVICE'] = torch.device(f'cuda:{gpu_id}')
        train_model(params)  # Your training function


if __name__ == '__main__':

    try:
        # Set start method to 'spawn' for compatibility
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # start method can only be set once, ignore if set before

    # Hyperparameters
    TILE_SIZE = 128
    OVERLAP = TILE_SIZE // 4
    PIC_BATCH_SIZE = 2
    BATCH_SIZE = 24
    EPOCHS = 4
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_MODEL = True
    DEPTH_PADDING = 2
    n_parallel = 1

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
        RangeTransform(in_range=(0, 2 ** 16), out_range=(0, 1)),
    ])

    learning_rate = [0.01]
    l1_lambda = [1]
    l2_lambda = [1]
    D_LR = [0.001]
    G_LR = [0.001]
    run_names = [
        "20240521-224927_Process-2",
        "20240521-224927_Process-3",
        "20240521-224927_Process-4",
        "20240521-224927_Process-5",
        "20240522-054727_Process-4",
        "20240522-054844_Process-2",
        "20240522-054932_Process-3",
        "20240522-054944_Process-5",
        "20240522-125503_Process-4",
        "20240522-125622_Process-2",
        "20240522-125643_Process-3",
        "20240522-125647_Process-5",
        "20240522-200743_Process-3",
        "20240522-200756_Process-5",
        "20240522-200758_Process-4",
        "20240522-200915_Process-2",
        "20240523-032321_Process-4",
        "20240523-032353_Process-3",
        "20240523-032513_Process-5",
        "20240523-032701_Process-2",
        "20240523-104612_Process-3",
        "20240523-104616_Process-4",
        "20240523-105000_Process-5",
        "20240523-105025_Process-2",
        "20240523-155215_Process-2",
        "20240523-155216_Process-3",
        "20240523-155216_Process-4",
        "20240523-155216_Process-5",
        "20240523-224928_Process-2",
        "20240523-224949_Process-4",
        "20240523-225035_Process-3",
        "20240523-225113_Process-5",
        "20240524-055236_Process-4",
        "20240524-055347_Process-2",
        "20240524-055451_Process-5",
        "20240524-055519_Process-3",
    ]

    run_names = [
        "20240523-032701_Process-2",
        "20240522-054944_Process-5",
        "20240523-155216_Process-4",
        "20240521-224927_Process-5"
    ]

    parameter_sets = []

    print("Loading dataset...")
    dataset = HarmonyDataset(
        root=data_dir,
        equalization="histogram",
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        transform=transform,
        depth_padding=DEPTH_PADDING,
        picture_batch_size=PIC_BATCH_SIZE,
        every_nth=2,
        start_nth=0,
    )
    print("Dataset loaded.")

    process = 0
    for run_name in run_names:
        for lr in learning_rate:
            for l1 in l1_lambda:
                for l2 in l2_lambda:
                    for g_lr in G_LR:
                        for d_lr in D_LR:
                            parameter_sets.append({
                                'Process': process,
                                'LEARNING_RATE': lr,
                                'DEPTH_PADDING': DEPTH_PADDING,
                                'L1_LAMBDA': l1,
                                'L2_LAMBDA': l2,
                                'loader': False,
                                'EPOCHS': EPOCHS,
                                'TRUE_BATCH_SIZE': TRUE_BATCH_SIZE,
                                'PIC_BATCH_SIZE': PIC_BATCH_SIZE,
                                'SAVE_MODEL': SAVE_MODEL,
                                'dataset': dataset,
                                'TILE_SIZE': TILE_SIZE,
                                'BATCH_SIZE': BATCH_SIZE,
                                "run_name": run_name,
                            })
                        process += 1

    print(f"Parameter sets: {len(parameter_sets)}")

    n_cuda = torch.cuda.device_count()
    n_workers = n_cuda * n_parallel
    n_parameter_sets = len(parameter_sets)

    # Setup multiprocessing

    manager = Manager()

    shared_param_sets = manager.list(parameter_sets)
    lock = manager.Lock()

    processes = []
    for i in range(n_workers):
        gpu_id = i % n_cuda  # Distribute GPUs among workers
        p = Process(target=worker_func, args=(shared_param_sets, lock, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
