import os
import time

import PIL
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DiscriminatorNetwork import DiscriminatorNetwork
from GeneratorNetwork import GeneratorNetwork, generate_full_test
from dataset import custom_collate_fn


class AsymmetricL2Loss(nn.Module):
    def __init__(self, underestimation_weight=2.0):
        """
        Initializes the AsymmetricL2Loss module.

        Parameters:
        underestimation_weight (float): The weight applied to the loss for underestimations.
                                        A value greater than 1.0 increases the penalty for underestimations.
        """
        super(AsymmetricL2Loss, self).__init__()
        self.underestimation_weight = underestimation_weight

    def forward(self, predictions, targets):
        """
        Forward pass of the loss function.

        Parameters:
        predictions (torch.Tensor): The predicted values from the model.
        targets (torch.Tensor): The actual target values.

        Returns:
        torch.Tensor: The computed asymmetric L2 loss.
        """
        # Calculate the difference between predictions and targets
        diff = predictions - targets

        # Apply a higher weight to negative differences (underestimations)
        weights = torch.where(diff < 0, self.underestimation_weight, 1.0)

        # Calculate the weighted L2 loss
        weighted_squared_diff = weights * (diff ** 2)
        loss = weighted_squared_diff.mean()  # Mean over all elements

        return loss


def train_model(training_params):

    run_name = training_params.get('run_name', None)

    import multiprocessing
    process = multiprocessing.current_process().name

    if run_name is None:
        print("No run name provided")
        time_stamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{time_stamp}_{process}"

    log_dir = f"runs_3/{run_name}"

    EPOCHS = int(training_params.get('EPOCHS', 10))

    try:
        param_file = None
        for file in os.listdir(f"{log_dir}/progress"):
            if "events.out.tfevents" in file:
                print(f"Loading previous parameters from {file}")
                param_file = file
                continue

            else:
                print(file)

        with open(f"{log_dir}/progress/{param_file}", 'rb') as f:
            for line in f.readlines():
                line = str(line)
                if "LEARNING_RATE" in line:
                    key_value_pairs = line.split("\\x01")[3].split("J")[0]
                    key_value_pairs = key_value_pairs.split(", ")
                    for pair in key_value_pairs:
                        key, value = pair.split(": ")
                        print(f"{key}: {value}")
                        training_params[key] = value
                    continue

    except FileNotFoundError:
        print("Didn't find previous parameters")

    LEARNING_RATE = float(training_params.get('LEARNING_RATE', 0.001))
    TILE_SIZE = int(training_params.get('TILE_SIZE', 128))
    DEPTH_PADDING = int(training_params.get('DEPTH_PADDING', 2))
    MIN_ENCODER_DIM = 16
    loader = training_params.get('loader', None)
    TRUE_BATCH_SIZE = int(training_params.get('TRUE_BATCH_SIZE', 32))
    PIC_BATCH_SIZE = int(training_params.get('PIC_BATCH_SIZE', 4))
    SAVE_MODEL = bool(training_params.get('SAVE_MODEL', False))
    L1_LAMBDA = float(training_params.get('L1_LAMBDA', 0.01))
    L2_LAMBDA = float(training_params.get('L2_LAMBDA', 0.01))
    DEVICE = training_params.get('DEVICE', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    G_LR = training_params.get('G_LR', LEARNING_RATE)
    D_LR = training_params.get('D_LR', LEARNING_RATE)

    if loader == False:
        dataset = training_params.get('dataset', None)
        BATCH_SIZE = TRUE_BATCH_SIZE // PIC_BATCH_SIZE

        if dataset is None:
            raise ValueError("No dataset provided")

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            collate_fn=custom_collate_fn,
            shuffle=True,
            num_workers=0
        )

    # test device
    x = torch.ones(1, device=DEVICE)
    print(x)

    # train model
    generator = GeneratorNetwork(out_channels=2,
                                 image_size=TILE_SIZE,
                                 depth_padding=DEPTH_PADDING,
                                 min_encoding_dim=MIN_ENCODER_DIM)
    discriminator = DiscriminatorNetwork(image_size=TILE_SIZE,
                                         depth_padding=DEPTH_PADDING)

    generator.to(DEVICE)
    discriminator.to(DEVICE)

    g_loss_fn = torch.nn.BCELoss()
    l1_loss_fn = torch.nn.L1Loss()
    weighted_l2 = AsymmetricL2Loss(underestimation_weight=2.0)
    d_loss_fn = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=G_LR)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=D_LR)

    try:
        generator.load_state_dict(torch.load(f"{log_dir}/generator.pt", map_location=DEVICE))
        discriminator.load_state_dict(torch.load(f"{log_dir}/discriminator.pt", map_location=DEVICE))
        print("Model loaded")
    except FileNotFoundError:
        print("Model not found")

    fake_writer = SummaryWriter(f"{log_dir}/fake")
    real_writer = SummaryWriter(f"{log_dir}/real")
    bf_writer = SummaryWriter(f"{log_dir}/brightfield")
    progress_writer = SummaryWriter(f"{log_dir}/progress")
    test_writer = SummaryWriter(f"{log_dir}/test")

    # Save parameters to tensorboard
    #progress_writer.add_text('Parameters',
    #                         f"LEARNING_RATE: {LEARNING_RATE}, TILE_SIZE: {TILE_SIZE}, DEPTH_PADDING: {DEPTH_PADDING}, MIN_ENCODER_DIM: {MIN_ENCODER_DIM}, EPOCHS: {EPOCHS}, TRUE_BATCH_SIZE: {TRUE_BATCH_SIZE}, PIC_BATCH_SIZE: {PIC_BATCH_SIZE}, SAVE_MODEL: {SAVE_MODEL}, L1_LAMBDA: {L1_LAMBDA}, L2_LAMBDA: {L2_LAMBDA}",
    #                         0)

    # Remove device from training_params
    training_params.pop('DEVICE', None)
    training_params.pop('loader', None)
    training_params.pop('dataset', None)

    #progress_writer.add_hparams(training_params,{'null': 0}, run_name=run_name, global_step=0)

    # load model


    # Extract test images
    iter_loader = iter(loader)
    bf_channels, true_fluorescent = next(iter_loader)
    test_bf_channels = bf_channels.to(DEVICE)
    test_true_fluorescent = true_fluorescent.to(DEVICE)
    indeces = np.arange(0, 4) * PIC_BATCH_SIZE
    dead_sample = test_true_fluorescent[indeces, 0]
    live_sample = test_true_fluorescent[indeces, 1]
    bf_sample = test_bf_channels[indeces, DEPTH_PADDING]

    dead_sample = dead_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
    dead_sample = torch.cat((dead_sample * 0.1, dead_sample * 0.8, dead_sample * 0), 1)

    live_sample = live_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
    live_sample = torch.cat((live_sample * 0.9, live_sample * 0.8, live_sample * 0), 1)

    bf_sample = bf_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
    #bf_sample = torch.cat((bf_sample, bf_sample, bf_sample), 1)

    print(bf_sample.shape)

    dead_real_grid = torchvision.utils.make_grid(dead_sample)
    live_real_grid = torchvision.utils.make_grid(live_sample)
    bf_real_grid = torchvision.utils.make_grid(bf_sample)

    print(dead_real_grid.shape)
    print(bf_real_grid.shape)

    test_writer.add_image('brightfield', bf_real_grid, 0)
    test_writer.add_image('live_fluorescent', live_real_grid, 0)
    test_writer.add_image('dead_fluorescent', dead_real_grid, 0)

    logging_steps = 0
    for epoch in range(EPOCHS):
        logging_time = 0
        print(f"Epoch {epoch}")

        if epoch != 0 and SAVE_MODEL:
            save_start_time = time.time()
            torch.save(generator.state_dict(), f"{log_dir}/generator.pt")
            torch.save(discriminator.state_dict(), f"{log_dir}/discriminator.pt")
            print(f"Model saved in {round(time.time() - save_start_time, 2)} seconds")
            # torch.save(generator.state_dict(), f"runs/generator_{time_stamp}.pt")
            # torch.save(discriminator.state_dict(), f"runs/discriminator_{time_stamp}.pt")

        for batch_idx, (bf_channels, true_fluorescent) in enumerate(loader):
            start_time = time.time()
            bf_channels = bf_channels.to(DEVICE)
            true_fluorescent = true_fluorescent.to(DEVICE)

            print(f"Batch {batch_idx}")
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            outputs = generator(bf_channels)

            disc_labels_true_labels = torch.ones((TRUE_BATCH_SIZE, 1)).to(DEVICE)

            disc_true_outputs = discriminator(bf_channels, true_fluorescent)
            disc_fake_outputs = discriminator(bf_channels, outputs)

            d_loss = (d_loss_fn(disc_true_outputs, disc_labels_true_labels) + d_loss_fn(disc_fake_outputs,
                                                                                        1 - disc_labels_true_labels)) / 2

            discriminator.zero_grad()

            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            disc_fake_outputs = discriminator(bf_channels, outputs)

            l1_loss_real = torch.nn.L1Loss()(outputs, true_fluorescent)
            l2_loss_real = torch.nn.MSELoss()(outputs, true_fluorescent)

            g_loss = g_loss_fn(disc_fake_outputs, disc_labels_true_labels) + \
                     L1_LAMBDA * l1_loss_real + \
                     L2_LAMBDA * l2_loss_real

            generator.zero_grad()

            g_loss.backward()
            g_optimizer.step()

            # Display one pair of real and generated images
            if time.time() - logging_time > 60:
                with torch.no_grad():
                    print(f"Time taken for batch {batch_idx}: {round(time.time() - start_time, 2)} seconds\n",
                          f"Discriminator loss: {d_loss.item()}, Generator loss: {g_loss.item()}"
                          )

                    logging_time = time.time()

                    # extract indeces for 4 different images

                    indeces = np.arange(0, 4) * PIC_BATCH_SIZE
                    dead_sample = test_true_fluorescent[indeces, 0]
                    live_sample = test_true_fluorescent[indeces, 1]
                    bf_sample = test_bf_channels[indeces, DEPTH_PADDING]

                    test_outputs = generator(bf_channels)

                    dead_sample_gen = test_outputs[indeces, 0]
                    live_sample_gen = test_outputs[indeces, 1]

                    # create channels to accommodate colors
                    dead_sample_gen = dead_sample_gen.view(-1, 1, TILE_SIZE, TILE_SIZE)
                    dead_sample_gen = torch.cat((dead_sample_gen * 0.1, dead_sample_gen * 0.8, dead_sample_gen * 0), 1)

                    live_sample_gen = live_sample_gen.view(-1, 1, TILE_SIZE, TILE_SIZE)
                    live_sample_gen = torch.cat((live_sample_gen * 0.9, live_sample_gen * 0.8, live_sample_gen * 0), 1)

                    dead_sample = dead_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
                    dead_sample = torch.cat((dead_sample * 0.1, dead_sample * 0.8, dead_sample * 0), 1)

                    live_sample = live_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)
                    live_sample = torch.cat((live_sample * 0.9, live_sample * 0.8, live_sample * 0), 1)

                    bf_sample = bf_sample.view(-1, 1, TILE_SIZE, TILE_SIZE)

                    dead_real_grid = torchvision.utils.make_grid(dead_sample)
                    live_real_grid = torchvision.utils.make_grid(live_sample)
                    bf_real_grid = torchvision.utils.make_grid(bf_sample)
                    dead_fake_grid = torchvision.utils.make_grid(dead_sample_gen)
                    live_fake_grid = torchvision.utils.make_grid(live_sample_gen)

                    real_writer.add_image('dead', dead_real_grid, logging_steps)
                    real_writer.add_image('live', live_real_grid, logging_steps)
                    bf_writer.add_image('input', bf_real_grid, logging_steps)
                    fake_writer.add_image('dead', dead_fake_grid, logging_steps)
                    fake_writer.add_image('live', live_fake_grid, logging_steps)

                    # log losses
                    progress_writer.add_scalar('Discriminator Loss', d_loss.item(), logging_steps)
                    progress_writer.add_scalar('Generator Loss', g_loss.item(), logging_steps)
                    progress_writer.add_scalar('Generator L1 Loss', l1_loss_real.item(), logging_steps)
                    progress_writer.add_scalar('Generator L2 Loss', l2_loss_real.item(), logging_steps)

                    # accuracy
                    disc_true_outputs = disc_true_outputs.detach().cpu().numpy()
                    disc_fake_outputs = disc_fake_outputs.detach().cpu().numpy()
                    disc_true_outputs = np.round(disc_true_outputs)
                    disc_fake_outputs = np.round(disc_fake_outputs)
                    true_accuracy = np.sum(disc_true_outputs) / len(disc_true_outputs)
                    fake_accuracy = np.sum(disc_fake_outputs) / len(disc_fake_outputs)
                    progress_writer.add_scalar('True Accuracy (% of real classified as real)', true_accuracy, logging_steps)
                    progress_writer.add_scalar('Fake Accuracy (% of fake classified as fake)', fake_accuracy, logging_steps)

                    if logging_steps % 10 == 0:
                        infer_start_time = time.time()
                        plot_buf = generate_full_test(dataset, TILE_SIZE, TILE_SIZE//4, DEVICE, generator)
                        image = PIL.Image.open(plot_buf)
                        image = torchvision.transforms.ToTensor()(image)
                        progress_writer.add_image('Full Test', image, logging_steps)
                        print(f"Time taken for inference: {round(time.time() - infer_start_time, 2)} seconds")


                    logging_steps += 1
    # test model
    # save model
    if SAVE_MODEL:
        torch.save(generator.state_dict(), f"{log_dir}/generator.pt")
        torch.save(discriminator.state_dict(), f"{log_dir}/discriminator.pt")
