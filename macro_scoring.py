import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torchmetrics.functional.image import structural_similarity_index_measure
from GeneratorNetwork import generate_full_test


def get_macro_scores(dataset, TILE_SIZE, OVERLAP, DEVICE, generator, subset=None):
    full_mse_dead = 0
    full_mse_live = 0
    full_mae_dead = 0
    full_mae_live = 0
    max_intensity_dead = 0
    max_intensity_live = 0
    ssim = structural_similarity_index_measure

    full_ssim_dead = 0
    full_ssim_live = 0

    wells = dataset.wells

    if subset is not None:
        wells = np.random.choice(wells, subset)

        print(f"Subset of {subset} wells selected.")
        print(f"Subset of wells: {wells}")


    with (torch.no_grad()):
        n_wells = len(wells)

        start = time.time()
        for well_idx, well in enumerate(wells):

            if well_idx % 10 == 0:
                print(f"Batch {well_idx}/{n_wells}")
                print(f"Percents: {round(well_idx / n_wells * 100)}%")
                elapsed = time.time() - start
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Time elapsed: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
                left = (time.time() - start) / (well_idx + 1) * (n_wells - well_idx)
                hours, remainder = divmod(left, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"Time left: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

            bf, dead, live, bf_real, dead_real, live_real = generate_full_test(
                dataset,
                TILE_SIZE,
                OVERLAP,
                DEVICE,
                generator,
                display=False,
                well=well,
                return_images=True,
            )

            plt.imshow(bf, cmap='gray')
            plt.axis('off')
            plt.show()
            plt.imshow(dead, cmap='Greens')
            plt.axis('off')
            plt.show()
            plt.imshow(live, cmap='Oranges')
            plt.axis('off')
            plt.show()
            plt.imshow(bf_real, cmap='gray')
            plt.axis('off')
            plt.show()
            plt.imshow(dead_real, cmap='Greens')
            plt.axis('off')
            plt.show()
            plt.imshow(live_real, cmap='Oranges')
            plt.axis('off')
            plt.show()

            # print(f"Generated fluorescent shape: {generated_fluorescent.shape}")
            # print(f"True fluorescent shape: {true_fluorescent.shape}")
            # print(f"Brightfield shape: {bf_channels.shape}")

            dead = torch.tensor(dead).reshape(1, 1, 1080, 1080)
            live = torch.tensor(live).reshape(1, 1, 1080, 1080)
            dead_real = torch.tensor(dead_real).reshape(1, 1, 1080, 1080)
            live_real = torch.tensor(live_real).reshape(1, 1, 1080, 1080)

            full_mse_dead += torch.nn.functional.mse_loss(dead / 255.0, dead_real / 255.0)
            full_mse_live += torch.nn.functional.mse_loss(live / 255.0, live_real / 255.0)

            full_mae_dead += torch.nn.functional.l1_loss(dead / 255.0, dead_real / 255.0)
            full_mae_live += torch.nn.functional.l1_loss(live / 255.0, live_real / 255.0)

            max_intensity_dead = torch.max(torch.tensor([max_intensity_dead, torch.max(dead / 255.0)]))
            max_intensity_live = torch.max(torch.tensor([max_intensity_live, torch.max(live / 255.0)]))

            full_ssim_dead += ssim(preds=dead / 255.0, target=dead_real / 255.0, data_range=(0.0, 1.0))
            full_ssim_live += ssim(preds=live / 255.0, target=live_real / 255.0, data_range=(0.0, 1.0))

        full_mse_dead = full_mse_dead / n_wells
        full_mse_live = full_mse_live / n_wells
        full_mae_dead = full_mae_dead / n_wells
        full_mae_live = full_mae_live / n_wells

        full_ssim_dead = full_ssim_dead / n_wells
        full_ssim_live = full_ssim_live / n_wells

        PSNR_dead = 20 * torch.log10(max_intensity_dead) - 10 * torch.log10(torch.tensor(full_mse_dead))
        PSNR_live = 20 * torch.log10(max_intensity_live) - 10 * torch.log10(torch.tensor(full_mse_live))

        return full_mse_dead, full_mse_live, full_mae_dead, full_mae_live, full_ssim_dead, full_ssim_live, PSNR_dead, PSNR_live, n_wells