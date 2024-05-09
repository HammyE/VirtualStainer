import subprocess

import matplotlib.pyplot as plt
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
    BATCH_SIZE = 8
    EPOCHS = 5
    MIN_ENCODER_DIM = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_MODEL = True
    DEPTH_PADDING = 2

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

    learning_rate = [0.01]
    l1_lambda = [0.1, 0.5]
    l2_lambda = [0.1]
    D_LR = [0.01]  # [0.001, 0.005, 0.01]
    G_LR = [0.001, 0.01]
    run_names = [  # "20240422-205546_Process-4",
        "20240423-165651_Process-3",
        "20240423-165649_Process-5",
        "20240423-165127_Process-4",
        "20240423-165122_Process-2",
        "20240423-115702_Process-5",
        "20240423-115653_Process-3",
        "20240423-115255_Process-4",
        "20240423-115245_Process-2",
        "20240423-065557_Process-5",
        "20240423-065549_Process-3",
        "20240423-065313_Process-4",
        "20240423-065303_Process-2",
        "20240423-015531_Process-5",
        "20240423-015528_Process-3",
        "20240423-015427_Process-4",
        "20240423-015425_Process-2",
        "20240422-205546_Process-5",
        "20240422-205546_Process-3",
        "20240422-205546_Process-2", ]

    run_names = [
        "20240508-171628_Process-2",
        "20240508-171628_Process-3",
        "20240509-043606_Process-2",
        "20240509-043802_Process-3",
        "20240509-164927_Process-2",
        "20240509-164927_Process-3",
        "20240422-205546_Process-3",
        "20240422-205546_Process-2"
    ]

    parameter_sets = []
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

    # Remove the first 4 parameter sets
    print(f"Parameter sets: {len(parameter_sets)}")

    n_cuda = torch.cuda.device_count()
    n_workers = n_cuda * 2
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
