import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from DiscriminatorNetwork import DiscriminatorNetwork
from GeneratorNetwork import GeneratorNetwork
from dataset import custom_collate_fn


def train_model(training_params):
    LEARNING_RATE = training_params.get('LEARNING_RATE', 0.001)
    TILE_SIZE = training_params.get('TILE_SIZE', 64)
    DEPTH_PADDING = training_params.get('DEPTH_PADDING', 2)
    MIN_ENCODER_DIM = 16
    EPOCHS = training_params.get('EPOCHS', 10)
    loader = training_params.get('loader', None)
    TRUE_BATCH_SIZE = training_params.get('TRUE_BATCH_SIZE', 32)
    PIC_BATCH_SIZE = training_params.get('PIC_BATCH_SIZE', 4)
    SAVE_MODEL = training_params.get('SAVE_MODEL', False)
    L1_LAMBDA = training_params.get('L1_LAMBDA', 0.01)
    L2_LAMBDA = training_params.get('L2_LAMBDA', 0.01)
    DEVICE = training_params.get('DEVICE', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    try:
        print(f"Training process {training_params['Process']}")
    except KeyError:
        pass

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
    d_loss_fn = torch.nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Add writers for tensorboard
    import multiprocessing
    process = multiprocessing.current_process().name

    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = f"runs/{time_stamp}_{process}"

    fake_writer = SummaryWriter(f"{log_dir}/fake")
    real_writer = SummaryWriter(f"{log_dir}/real")
    bf_writer = SummaryWriter(f"{log_dir}/brightfield")
    progress_writer = SummaryWriter(f"{log_dir}/progress")

    # Save parameters to tensorboard
    progress_writer.add_text('Parameters', f"LEARNING_RATE: {LEARNING_RATE}, TILE_SIZE: {TILE_SIZE}, DEPTH_PADDING: {DEPTH_PADDING}, MIN_ENCODER_DIM: {MIN_ENCODER_DIM}, EPOCHS: {EPOCHS}, TRUE_BATCH_SIZE: {TRUE_BATCH_SIZE}, PIC_BATCH_SIZE: {PIC_BATCH_SIZE}, SAVE_MODEL: {SAVE_MODEL}, L1_LAMBDA: {L1_LAMBDA}, L2_LAMBDA: {L2_LAMBDA}", 0)

    logging_steps = 0
    for epoch in range(EPOCHS):
        logging_time = 0
        print(f"Epoch {epoch}")

        if epoch != 0 and SAVE_MODEL:
            save_start_time = time.time()
            torch.save(generator.state_dict(), f"{log_dir}/generator.pt")
            torch.save(discriminator.state_dict(), f"{log_dir}/discriminator.pt")
            print(f"Model saved in {round(time.time() - save_start_time, 2)} seconds")
            torch.save(generator.state_dict(), f"runs/generator_{time_stamp}.pt")
            #torch.save(discriminator.state_dict(), f"runs/discriminator_{time_stamp}.pt")

        for batch_idx, (bf_channels, true_fluorescent) in enumerate(loader):
            start_time = time.time()
            bf_channels = bf_channels.to(DEVICE)
            true_fluorescent = true_fluorescent.to(DEVICE)

            print(f"Batch {batch_idx}")
            if epoch == 0 and batch_idx == 0:
                print(bf_channels.shape)
                # print(true_fluorescent.shape)
                # # calculate the memory usage of a batch
                # print("Memory usage of input: ", bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024,
                #       "MB")
                # print("Memory usage of output: ",
                #       true_fluorescent.element_size() * true_fluorescent.nelement() / 1024 / 1024, "MB")
                # print("Memory usage of both: ",
                #       bf_channels.element_size() * bf_channels.nelement() / 1024 / 1024 + true_fluorescent.element_size() * true_fluorescent.nelement() / 1024 / 1024,
                #       "MB")

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

            g_loss = (g_loss_fn(disc_fake_outputs, disc_labels_true_labels) +
                      L1_LAMBDA * torch.nn.L1Loss()(outputs, true_fluorescent) +
                      L2_LAMBDA * torch.nn.MSELoss()(outputs, true_fluorescent))

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
                dead_sample = true_fluorescent[indeces, 0]
                live_sample = true_fluorescent[indeces, 1]
                bf_sample = bf_channels[indeces, DEPTH_PADDING]

                dead_sample_gen = outputs[indeces, 0]
                live_sample_gen = outputs[indeces, 1]

                # create channels to accomodate colors
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

                # accuracy
                disc_true_outputs = disc_true_outputs.detach().cpu().numpy()
                disc_fake_outputs = disc_fake_outputs.detach().cpu().numpy()
                disc_true_outputs = np.round(disc_true_outputs)
                disc_fake_outputs = np.round(disc_fake_outputs)
                true_accuracy = np.sum(disc_true_outputs) / len(disc_true_outputs)
                fake_accuracy = np.sum(disc_fake_outputs) / len(disc_fake_outputs)
                progress_writer.add_scalar('True Accuracy', true_accuracy, logging_steps)
                progress_writer.add_scalar('Fake Accuracy', fake_accuracy, logging_steps)

                logging_steps += 1
    # test model
    # save model
    if SAVE_MODEL:
        torch.save(generator.state_dict(), f"{log_dir}/generator.pt")
        torch.save(discriminator.state_dict(), f"{log_dir}/discriminator.pt")
