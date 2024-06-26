'''This is the main file of the project. It will be used to run the project.'''
import os
import subprocess
import time
import sys

import PIL
import numpy as np
import torch
import torchvision
# Importing the necessary modules
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt

from DiscriminatorNetwork import DiscriminatorNetwork
from GeneratorNetwork import GeneratorNetwork, generate_full_test
from MaximumIntensityProjection import MaximumIntensityProjection
from RangeTransform import RangeTransform
from dataset import HarmonyDataset, custom_collate_fn
from torchmetrics.functional.image import structural_similarity_index_measure

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
    OVERLAP = TILE_SIZE // 4
    PIC_BATCH_SIZE = 2
    BATCH_SIZE = 78
    EPOCHS = 0
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
        RangeTransform(in_range=(0, 2**16), out_range=(0, 1)),
    ])

    print("Loading dataset...")

    disallowed_datasets = [
        #'2307130102__2023-07-23T08_50_42-Measurement 11',
        #'2307130202__2023-07-23T10_32_16-Measurement 11',
        #'2307130302__2023-07-23T12_13_49-Measurement 11'
    ]

    dataset = HarmonyDataset(
        root=data_dir,
        equalization="histogram",
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        transform=transform,
        depth_padding=DEPTH_PADDING,
        picture_batch_size=PIC_BATCH_SIZE,
        every_nth=2,
        start_nth=1,
    )

    print("Dataset loaded with length: ", len(dataset))

    print("Loading data loader...")
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    print("Data loader loaded.")

    if DEBUG:
        display_images()

    if EPOCHS > 0:
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
             'D_LR': D_LR,
             'run_name': "20240417-000820_Process-5"}
        )

    # load model

    model_paths = """20240522-054727_Process-4""".split('\n')

    if not os.path.exists('results'):
        os.mkdir('results')


    for model_path in model_paths:
        if not os.path.exists(f'results/{model_path}'):
            os.mkdir(f'results/{model_path}')
            os.mkdir(f'results/{model_path}/images')

        img_path = f'results/{model_path}/images'
        result_path = f'results/{model_path}/data.csv'
        model_path = f'runs_10/{model_path}'

        generator = GeneratorNetwork(out_channels=2, image_size=TILE_SIZE, depth_padding=DEPTH_PADDING,
                                     min_encoding_dim=MIN_ENCODER_DIM).to(DEVICE)
        generator.load_state_dict(torch.load(f'{model_path}/generator.pt', map_location=DEVICE))

        # Show the network architecture
        print(
            f"Generator network architecture:\n{generator}\nNumber of parameters: {sum(p.numel() for p in generator.parameters())}")


        #generate_full_test(dataset, TILE_SIZE, OVERLAP, DEVICE, generator, display=not torch.cuda.is_available())

        full_mse = 0
        full_mae = 0
        max_i = 0
        full_ssim = 0
        ssim = structural_similarity_index_measure

        live_mse = 0
        live_mae = 0
        live_ssim = 0
        live_max_i = 0

        dead_mse = 0
        dead_mae = 0
        dead_ssim = 0
        dead_max_i = 0




        with (torch.no_grad()):
            n_batches = len(loader) // TRUE_BATCH_SIZE
            start = time.time()
            for batch_idx, (bf_channels, true_fluorescent) in enumerate(loader):



                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(loader)}")
                    print(f"Percents: {round(batch_idx / len(loader) * 100)}%")
                    elapsed = time.time() - start
                    hours, remainder = divmod(elapsed, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                    left = (time.time() - start) / (batch_idx + 1) * (len(loader) - batch_idx)
                    hours, remainder = divmod(left, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    print(f"Time left: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                    buffer = generate_full_test(dataset, TILE_SIZE, OVERLAP, DEVICE, generator,display=False)
                    image = PIL.Image.open(buffer)
                    image.save(f'{img_path}/image_{batch_idx}.png')
                bf_channels = bf_channels.to(DEVICE)
                true_fluorescent = true_fluorescent.to(DEVICE)
                generated_fluorescent = generator(bf_channels)


                if batch_idx == -1:
                    for img_idx in range(4):
                        input_1 = bf_channels[img_idx]
                        output_1 = generated_fluorescent[img_idx]
                        true_f = true_fluorescent[img_idx]

                        #for i in range(5):
                        #    plt.imshow(input_1[i].cpu().numpy(), cmap='gray')
                        #    plt.show()

                        plt.subplot(4, 4, 4*img_idx + 1)
                        plt.imshow(output_1[0].cpu().numpy(), cmap='Greens')
                        plt.axis('off')
                        plt.subplot(4, 4, 4 * img_idx + 2)
                        plt.imshow(true_f[0].cpu().numpy(), cmap='Greens')
                        plt.axis('off')
                        plt.subplot(4, 4, 4 * img_idx + 3)
                        #plt.show()
                        plt.imshow(output_1[1].cpu().numpy(), cmap='Oranges')
                        plt.axis('off')
                        plt.subplot(4, 4, 4 * img_idx + 4)
                        plt.imshow(true_f[1].cpu().numpy(), cmap='Oranges')
                        plt.axis('off')
                    plt.show()

                print(f"Generated fluorescent max: {torch.max(generated_fluorescent)}")

                full_mse += torch.nn.functional.mse_loss(generated_fluorescent, true_fluorescent) * bf_channels.shape[0]
                full_mae += torch.nn.functional.l1_loss(generated_fluorescent, true_fluorescent) * bf_channels.shape[0]
                max_i = torch.max(torch.tensor([max_i, torch.max(generated_fluorescent)]))
                full_ssim += ssim(preds=generated_fluorescent, target=true_fluorescent, data_range=(0.0, 1.0)) * bf_channels.shape[0]

                live_pred = generated_fluorescent[:, 1]
                dead_pred = generated_fluorescent[:, 0]
                live_target = true_fluorescent[:, 1]
                dead_target = true_fluorescent[:, 0]

                current_batch_size = bf_channels.shape[0]


                live_pred = live_pred.reshape(current_batch_size, 1, TILE_SIZE, TILE_SIZE)
                dead_pred = dead_pred.reshape(current_batch_size, 1, TILE_SIZE, TILE_SIZE)
                live_target = live_target.reshape(current_batch_size, 1, TILE_SIZE, TILE_SIZE)
                dead_target = dead_target.reshape(current_batch_size, 1, TILE_SIZE, TILE_SIZE)

                live_mse += torch.nn.functional.mse_loss(live_pred, live_target) * bf_channels.shape[0]
                live_mae += torch.nn.functional.l1_loss(live_pred, live_target) * bf_channels.shape[0]
                live_max_i = torch.max(torch.tensor([live_max_i, torch.max(live_pred)]))
                live_ssim += ssim(preds=live_pred, target=live_target, data_range=(0.0, 1.0)) * bf_channels.shape[0]

                dead_mse += torch.nn.functional.mse_loss(dead_pred, dead_target) * bf_channels.shape[0]
                dead_mae += torch.nn.functional.l1_loss(dead_pred, dead_target) * bf_channels.shape[0]
                dead_max_i = torch.max(torch.tensor([dead_max_i, torch.max(dead_pred)]))
                dead_ssim += ssim(preds=dead_pred, target=dead_target, data_range=(0.0, 1.0)) * bf_channels.shape[0]


                print(f"MSE: {torch.nn.functional.mse_loss(generated_fluorescent, true_fluorescent)}")
                print(f"MAE: {torch.nn.functional.l1_loss(generated_fluorescent, true_fluorescent)}")
                print(f"Images: {bf_channels.shape[0]*batch_idx} / {len(dataset)}")
                # print(f"SSIM: {ssim(preds=generated_fluorescent, target=true_fluorescent, data_range=(0.0,1.0))}")


            mse = full_mse / len(dataset)
            mae = full_mae / len(dataset)
            ssim = full_ssim / len(dataset)
            PSNR = 20 * torch.log10(max_i) - 10 * torch.log10(torch.tensor(mse))

            live_mse = live_mse / len(dataset)
            live_mae = live_mae / len(dataset)
            live_ssim = live_ssim / len(dataset)
            live_PSNR = 20 * torch.log10(live_max_i) - 10 * torch.log10(torch.tensor(live_mse))

            dead_mse = dead_mse / len(dataset)
            dead_mae = dead_mae / len(dataset)
            dead_ssim = dead_ssim / len(dataset)
            dead_PSNR = 20 * torch.log10(dead_max_i) - 10 * torch.log10(torch.tensor(dead_mse))

            print(f"Full MSE: {mse}")
            print(f"Full MAE: {mae}")
            print(f"Full SSIM: {ssim}")
            print(f"PSNR: {PSNR}")

            # Save the data
            with open(result_path, 'w') as f:
                f.write(f"Measure,Value\n")
                f.write(f"MSE,{mse}\n")
                f.write(f"MAE,{mae}\n")
                f.write(f"SSIM,{ssim}\n")
                f.write(f"PSNR,{PSNR}\n")
                f.write(f"Live MSE,{live_mse}\n")
                f.write(f"Live MAE,{live_mae}\n")
                f.write(f"Live SSIM,{live_ssim}\n")
                f.write(f"Live PSNR,{live_PSNR}\n")
                f.write(f"Dead MSE,{dead_mse}\n")
                f.write(f"Dead MAE,{dead_mae}\n")
                f.write(f"Dead SSIM,{dead_ssim}\n")
                f.write(f"Dead PSNR,{dead_PSNR}\n")
                f.write(f"Model path,{model_path}\n")
                f.write(f"n images,{len(dataset)}\n")
