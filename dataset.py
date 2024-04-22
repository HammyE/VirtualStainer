import os
import torch
import cv2
import numpy as np
# from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from torch.utils.data import Dataset
from MaximumIntensityProjection import equalize, get_equalization_params, get_blur, gaussian
from TileTransform import TileTransform

import matplotlib.pyplot as plt


class ImageSample():

    def __init__(self, well, depth, center, measurement):
        self.well = well
        self.depth = depth
        self.center = center
        self.measurement = measurement

    def __str__(self):
        return f"Well: {self.well}, Depth: {self.depth}, Center: {self.center}, Measurement: {self.measurement}"

    def __repr__(self):
        return f"Well: {self.well}, Depth: {self.depth}, Center: {self.center}, Measurement: {self.measurement}"


class HarmonyDataset(Dataset):
    def __init__(self, root, equalization=None, tile_size=256, overlap=32, depth_padding=1, depth_range=20,
                 picture_batch_size=8, transform=None):
        '''
        This is dataset class for harmony dataset, that parses the file structure and returns the brightfield images
        and the corresponding stained images.
        :param root: str, root directory of the dataset
        :param equalization: str, equalization method to be used (clahe, histogram, linear or None)
        :param transform: torchvision.Transforms, transform to be applied to the images
        '''
        self.main_dir = root
        self.transform = transform
        self.bf_stacks = {}
        self.dead_stacks = {}
        self.live_stacks = {}
        self.flourescent_channels = {}
        self.equalization = equalization
        self.root = root
        self.wells = []
        self.debug = False
        self.image_size = None
        self.depth_padding = depth_padding
        self.tile_size = tile_size
        self.depth_range = depth_range
        self.picture_batch_size = picture_batch_size

        # self.max_intensity = MaximumIntensityProjection(equalization_method=equalization)
        self.tile_transform = TileTransform(tile_size=tile_size, overlap=overlap)
        self.equalization_params_brightfield = {}
        self.equalization_params_dead = {}
        self.equalization_params_live = {}

        self.depths = {}
        self.masks = {}
        self.potential_centers = {}

        self.initial_guess = [1, 1, 1]

        self.len = None

        measurements = os.listdir(root)

        # Load every measurement
        for index, measurement in enumerate(measurements):

            if measurement.contains("DS_Store"):
                continue

            # Load or create equalization params, and cache them
            self.load_equalization(measurement)

            images = os.listdir(os.path.join(root, measurement, "images"))

            # Extract the wells from the images
            if self.debug: print(f"Extracting wells from measurement {measurement}...")
            plate_wells = self.extract_wells(images, measurement)

            # set image size
            self.image_size = cv2.imread(self.bf_stacks[self.wells[0]][0], cv2.IMREAD_GRAYSCALE).shape[0]

            min_pos = tile_size / 2 + 1
            max_pos = self.image_size - tile_size / 2 - 1
            # define in focus range for each well
            if self.debug: print(f"Generating samples for measurement {measurement}...")
            self.generate_samples(plate_wells, depth_padding, max_pos, measurement, min_pos)

            break

    def generate_samples(self, plate_wells, depth_padding, max_pos, measurement, min_pos):
        for well in plate_wells:
            self.find_depth_and_mask(measurement, well)

            if not os.path.isfile(os.path.join(self.root, measurement, "potential_centers", well + ".npy")):
                if not os.path.exists(os.path.join(self.root, measurement, "potential_centers")):
                    os.makedirs(os.path.join(self.root, measurement, "potential_centers"))
                mask = cv2.imread(self.masks[well], cv2.IMREAD_GRAYSCALE)
                potential_centers = np.argwhere(mask > 0)
                # remove potential centers that are too close to the edge
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 0] < min_pos), axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 0] > max_pos), axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 1] < min_pos), axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 1] > max_pos), axis=0)

                np.save(os.path.join(self.root, measurement, "potential_centers", well + ".npy"), potential_centers)

            self.potential_centers[well] = os.path.join(self.root, measurement, "potential_centers", well + ".npy")

            min_depth = self.depths[well][0]
            max_depth = self.depths[well][-1]
            if min_depth - depth_padding < 0:
                diff = depth_padding - min_depth
                min_depth = depth_padding
                max_depth += diff
            if max_depth + depth_padding >= len(self.bf_stacks[well]):
                diff = max_depth - len(self.bf_stacks[well]) + depth_padding + 1
                max_depth = len(self.bf_stacks[well]) - depth_padding - 1
                min_depth -= diff

            try:
                # check if the depths are within the range of the bf stack
                assert min_depth - self.depth_padding >= 0
                assert max_depth + self.depth_padding < len(self.bf_stacks[well])
                assert len(np.arange(min_depth, max_depth + 1)) == self.depth_range
            except AssertionError as e:
                print(f"Index error: {well}, {min_depth}, {max_depth}")
                print(f"Depth: {self.depths[well]}")
                print(f"Stack: {self.bf_stacks[well]}")
                raise e

            # change depths to new min and max
            self.depths[well] = np.arange(min_depth, max_depth + 1)

    def find_depth_and_mask(self, measurement, well):

        # Check if depths and masks already exist
        if os.path.isfile(os.path.join(self.root, measurement, "masks", well + ".tiff")):
            self.masks[well] = os.path.join(self.root, measurement, "masks", well + ".tiff")
            if self.debug: print(f"Mask for well {well} already exists.")
        else:
            if self.debug: print(f"Finding potential centers for well {well}...")
            final_mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            for level in self.bf_stacks[well]:
                img = cv2.imread(level, cv2.IMREAD_GRAYSCALE)
                img = equalize(img, self.equalization_params_brightfield[measurement])
                mask = self.tile_transform.get_mask(img)
                final_mask = np.max([final_mask, mask], axis=0)

            potential_centers = np.argwhere(final_mask > 0)

            # save potential centers to hard drive
            if not os.path.exists(os.path.join(self.root, measurement, "potential_centers")):
                os.makedirs(os.path.join(self.root, measurement, "potential_centers"))
            np.save(os.path.join(self.root, measurement, "potential_centers", well + ".npy"), potential_centers)

            # Save the mask to hard drive
            if not os.path.exists(os.path.join(self.root, measurement, "masks")):
                os.makedirs(os.path.join(self.root, measurement, "masks"))
            cv2.imwrite(os.path.join(self.root, measurement, "masks", well + ".tiff"), final_mask)

            self.potential_centers[well] = os.path.join(self.root, measurement, "potential_centers", well + ".npy")
            self.masks[well] = os.path.join(self.root, measurement, "masks", well + ".tiff")

        if os.path.isfile(os.path.join(self.root, measurement, "depths", well + ".npy")):
            self.depths[well] = np.load(os.path.join(self.root, measurement, "depths", well + ".npy"))
            if self.debug: print(f"Depths for well {well} already exist.")
        else:
            flourescent_intensity_list = []
            blur_list = []
            mask = cv2.imread(self.masks[well], cv2.IMREAD_GRAYSCALE)
            for img in self.bf_stacks[well]:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                img = equalize(img, self.equalization_params_brightfield[measurement])
                blur_list.append(get_blur(img, mask))

            for dead, live in zip(self.dead_stacks[well], self.live_stacks[well]):
                dead_img = cv2.imread(dead, cv2.IMREAD_GRAYSCALE)
                live_img = cv2.imread(live, cv2.IMREAD_GRAYSCALE)
                dead_img = equalize(dead_img, self.equalization_params_dead[measurement])
                live_img = equalize(live_img, self.equalization_params_live[measurement])
                flourescent_intensity_list.append(np.max([np.mean(live_img), np.mean(dead_img)]))
                #flourescent_intensity_list.append(np.mean(live_img))

            # Fit the curve to a gaussian curve
            x = np.arange(0, len(blur_list), 1)
            y = np.array(blur_list)
            y = np.array(flourescent_intensity_list)
            # Fit the curve to a gaussian curve
            try:
                popt, pcov = curve_fit(gaussian, x, y, p0=[1, 1, 1])
                popt, pcov = curve_fit(gaussian, x, flourescent_intensity_list, p0=[1, 1, 1])
            except:

                max_y = np.max(y)
                index_x = np.argmax(y)
                popt = [max_y, index_x, self.initial_guess[2]]
                if self.debug:
                    plt.scatter(x, y)
                    plt.show()

            # Get the 20 images with the highest focus according to the gaussian curve
            self.depths[well] = sorted(np.argsort(gaussian(x, *popt))[-self.depth_range:])

            # store the depths to hard drive
            if not os.path.exists(os.path.join(self.root, measurement, "depths")):
                os.makedirs(os.path.join(self.root, measurement, "depths"))
            np.save(os.path.join(self.root, measurement, "depths", well + ".npy"), self.depths[well])

            if self.debug:
                if self.debug: print(f"Plotting focus curve for well {well}...")
                if self.debug: print(f"mu: {popt[1]}")
                if self.debug: print(f"sigma: {popt[2]}")
                if self.debug: print(f"amplitude: {popt[0]}")

                plt.plot(x, y, 'b-', label='data')
                plt.plot(x, gaussian(x, *popt), 'r-', label='fit')

                # add lines for the 20 images with the highest focus
                min = self.depths[well][0]
                max = self.depths[well][-1]
                plt.axvline(x=min, color='g', linestyle='--')
                plt.axvline(x=max, color='g', linestyle='--')
                plt.title(well)

                # add ticks for min and max
                plt.xticks([min, max])

                plt.show()
                if self.debug: print(self.depths[well])

                # Show 16 images with the highest focus, evenly spaced in the list
                n_layers = len(self.bf_stacks[well])
                n_images_row = 4
                n_rows = 4
                fig, axs = plt.subplots(n_rows, n_images_row, figsize=(15, 15))
                idx = 0

                left = self.image_size
                right = 0
                top = self.image_size
                bottom = 0
                mask = cv2.imread(self.masks[well], cv2.IMREAD_GRAYSCALE)
                for pixel in np.argwhere(mask > 0):
                    left = np.min([left, int(pixel[0])])
                    right = np.max([right, int(pixel[0])])
                    top = np.min([top, int(pixel[1])])
                    bottom = np.max([bottom, int(pixel[1])])

                size = np.max([right - left, bottom - top])

                top = bottom - size
                left = right - size

                try:
                    for i in range(0, n_layers, int(n_layers / 16)):
                        img = cv2.imread(self.bf_stacks[well][i], cv2.IMREAD_GRAYSCALE)
                        dead = cv2.imread(self.dead_stacks[well][i], cv2.IMREAD_GRAYSCALE)
                        live = cv2.imread(self.live_stacks[well][i], cv2.IMREAD_GRAYSCALE)

                        dead = equalize(dead, self.equalization_params_dead[measurement])
                        live = equalize(live, self.equalization_params_live[measurement])

                        img = equalize(img, self.equalization_params_brightfield[measurement])
                        img = 255 - img

                        img = np.zeros_like(img)

                        # join the images
                        img = np.stack([dead, live, img], axis=2)

                        img = img[left:right, top:bottom]
                        ax = axs[idx // n_images_row, idx % n_images_row]
                        ax.imshow(img, vmin=0, vmax=1000)#, cmap='gray')
                        ax.axis('off')
                        ax.title.set_text(f"Layer {i}")
                        idx += 1
                except:
                    pass
                plt.show()

    def extract_wells(self, images, measurement):
        plate_wells = []
        for image in images:
            if image.endswith("f01p01-ch1sk1fk1fl1.tiff"):
                dead_channels = [os.path.join(self.root, measurement, "images", image)]
                live_channels = [os.path.join(self.root, measurement, "images",
                                              image.replace("-ch1sk1fk1fl1.tiff", "-ch2sk1fk1fl1.tiff"))]
                bf_channels = [os.path.join(self.root, measurement, "images",
                                            image.replace("-ch1sk1fk1fl1.tiff", "-ch3sk1fk1fl1.tiff"))]
                plane = 1
                while True:
                    plane += 1
                    if plane < 10:
                        plane_str = "0" + str(plane)
                    else:
                        plane_str = str(plane)

                    # check if the file exists
                    if not os.path.isfile(os.path.join(self.root, measurement, "images",
                                                       image.replace("f01p01-ch1sk1fk1fl1.tiff",
                                                                     "f01p" + plane_str + "-ch1sk1fk1fl1.tiff"))):
                        break

                    dead_channels.append(
                        os.path.join(
                            self.root,
                            measurement,
                            "images",
                            image.replace(
                                "f01p01-ch1sk1fk1fl1.tiff",
                                "f01p" + plane_str + "-ch1sk1fk1fl1.tiff")
                        )
                    )
                    live_channels.append(
                        os.path.join(
                            self.root,
                            measurement,
                            "images",
                            image.replace(
                                "f01p01-ch1sk1fk1fl1.tiff",
                                "f01p" + plane_str + "-ch2sk1fk1fl1.tiff")
                        )
                    )
                    bf_channels.append(
                        os.path.join(
                            self.root,
                            measurement,
                            "images",
                            image.replace(
                                "f01p01-ch1sk1fk1fl1.tiff",
                                "f01p" + plane_str +
                                "-ch3sk1fk1fl1.tiff")
                        )
                    )

                well_name = measurement + image.split("-")[0]

                self.wells.append(well_name)

                plate_wells.append(well_name)

                self.bf_stacks[well_name] = (bf_channels)
                self.dead_stacks[well_name] = (dead_channels)
                self.live_stacks[well_name] = (live_channels)
        return plate_wells

    def load_equalization(self, measurement):
        if os.path.exists(os.path.join(self.root, measurement, "equalization")):
            if self.debug: print("Equalization folder exists.")
        else:
            os.makedirs(os.path.join(self.root, measurement, "equalization"))
            if self.debug: print("Equalization folder created.")
        plate = os.path.dirname(os.path.join(self.root, measurement, "images")).replace("dataset/", "").replace(
            "/images", "")
        if os.path.isfile(os.path.join(self.root, measurement, "equalization", "equalization_params_brightfield.npy")):
            if self.debug: print("Equalization params already exist.")

            bf_params = np.load(
                os.path.join(self.root, measurement, "equalization", "equalization_params_brightfield.npy"),
                allow_pickle=False)
            dead_params = np.load(os.path.join(self.root, measurement, "equalization", "equalization_params_dead.npy"),
                                  allow_pickle=False)
            live_params = np.load(os.path.join(self.root, measurement, "equalization", "equalization_params_live.npy"),
                                  allow_pickle=False)

            self.equalization_params_brightfield[plate] = bf_params
            self.equalization_params_dead[plate] = dead_params
            self.equalization_params_live[plate] = live_params


        else:
            # get params and save them
            if self.debug: print("Equalization params do not exist.")
            if self.debug: print("Getting equalization params...")
            all_images_brightfield = []
            all_images_dead = []
            all_images_live = []
            for img in os.listdir(os.path.join(self.main_dir, plate, "images")):

                if "ch3" in img:
                    img = os.path.join(self.main_dir, plate, "images", img)
                    all_images_brightfield.append(img)
                    all_images_dead.append(img.replace("ch3", "ch1"))
                    all_images_live.append(img.replace("ch3", "ch2"))

            # load images
            self.equalization_params_brightfield[plate] = get_equalization_params(all_images_brightfield)
            self.equalization_params_dead[plate] = get_equalization_params(all_images_dead, [0.01, 0.999])
            self.equalization_params_live[plate] = get_equalization_params(all_images_live, [0.01, 0.999])

            if self.debug: print("Saving equalization params...")
            bf_params = np.array(self.equalization_params_brightfield[plate])
            dead_params = np.array(self.equalization_params_dead[plate])
            live_params = np.array(self.equalization_params_live[plate])
            np.save(os.path.join(self.root, measurement, "equalization", "equalization_params_brightfield.npy"),
                    bf_params)
            np.save(os.path.join(self.root, measurement, "equalization", "equalization_params_dead.npy"), dead_params)
            np.save(os.path.join(self.root, measurement, "equalization", "equalization_params_live.npy"), live_params)

            if self.debug: print("Equalization params saved.")

    def __len__(self):
        """
        This method returns the length of the dataset. This is the number of wells in the dataset times the number of
        depths in each well.
        :return:
        """
        if self.len is not None:
            return self.len
        self.len = len(self.wells) * self.depth_range
        return self.len

    def __getitem__(self, idx):

        try:

            well_idx = idx // self.depth_range
            depth_idx = idx % self.depth_range

            if self.debug: print(f"Getting well... {self.wells[well_idx]}")
            try:
                if self.debug: print(f"Getting depth... {self.depths[self.wells[well_idx]][depth_idx]}")
            except Exception as e:
                if self.debug: print(f"Depth not found for well {self.wells[well_idx]}")
                if self.debug: print(f"Depth index: {depth_idx}")
                if self.debug: print(f"Depths: {self.depths[self.wells[well_idx]]}")
                raise e

            well = self.wells[well_idx]

            # To make sure that all the depths of a well are used before repeating
            depth = self.depths[well][depth_idx]

            measurement = well.split("f")[0][:-6]

            x = torch.tensor(
                np.zeros((self.picture_batch_size, self.depth_padding * 2 + 1, self.tile_size, self.tile_size)))
            y = torch.tensor(np.zeros((self.picture_batch_size, 2, self.tile_size, self.tile_size)))

            bf_images = []

            skipped_tiles = 0

            if self.debug:
                plt_pic = 0
                plt.figure(figsize=(10, 20))

            for i in range(depth - self.depth_padding, depth + self.depth_padding + 1):
                try:
                    bf_img = cv2.imread(self.bf_stacks[well][i], cv2.IMREAD_GRAYSCALE)
                    bf_img = equalize(bf_img, self.equalization_params_brightfield[measurement])
                    bf_images.append(bf_img)
                except IndexError as e:
                    for img in self.bf_stacks[well]:
                        print(img)
                    print(f"Index error: {well}, {i}")
                    print(f"Depth: {depth}")
                    raise e

            dead_img = cv2.imread(self.dead_stacks[well][depth], cv2.IMREAD_GRAYSCALE)
            live_img = cv2.imread(self.live_stacks[well][depth], cv2.IMREAD_GRAYSCALE)

            dead_img = equalize(dead_img, self.equalization_params_dead[measurement])
            live_img = equalize(live_img, self.equalization_params_live[measurement])

            mask = cv2.imread(self.masks[well], cv2.IMREAD_GRAYSCALE)
            try:
                potential_centers = np.load(self.potential_centers[well])
            except Exception as e:
                print("Recreating potential centers")
                min_pos = self.tile_size / 2 + 1
                max_pos = self.image_size - self.tile_size / 2 - 1

                potential_centers = np.argwhere(mask > 0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 0] < min_pos),
                                              axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 0] > max_pos),
                                              axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 1] < min_pos),
                                              axis=0)
                potential_centers = np.delete(potential_centers, np.argwhere(potential_centers[:, 1] > max_pos),
                                              axis=0)
                np.save(os.path.join(self.root, measurement, "potential_centers", well + ".npy"), potential_centers)

            for picture in range(self.picture_batch_size):
                valid_tile = False
                while valid_tile == False:
                    center_idx = np.random.randint(0, len(potential_centers))
                    center = potential_centers[center_idx]

                    left = center[0] - round(self.tile_size / 2)
                    right = center[0] + round(self.tile_size / 2)
                    top = center[1] - round(self.tile_size / 2)
                    bottom = center[1] + round(self.tile_size / 2)

                    try:

                        if np.quantile(mask[left:right, top:bottom], 0.3) == 0:
                            skipped_tiles += 1
                            potential_centers = np.delete(potential_centers, center_idx, axis=0)
                        else:
                            valid_tile = True
                    except IndexError as e:
                        print(f"Index error: {well}, {depth}")
                        print(f"Depth: {depth}")
                        print(self.masks[well])
                        print(center)
                        print(left, right, top, bottom)

                        plt.imshow(mask)
                        plt.scatter(center[1], center[0], c='r', s=10, marker='x')
                        # draw bounding box
                        plt.plot([top, top], [right, left], 'b', linewidth=0.5)
                        plt.plot([top, bottom], [left, left], 'b', linewidth=0.5)
                        plt.plot([bottom, bottom], [left, right], 'b', linewidth=0.5)
                        plt.plot([bottom, top], [right, right], 'b', linewidth=0.5)
                        plt.savefig(f"error_{idx}.png")

                        raise e

                horizontal_flip = np.random.rand() > 0.5
                vertical_flip = np.random.rand() > 0.5
                rotation = np.random.randint(0, 4)

                for i, bf_img in enumerate(bf_images):
                    bf_tile = torch.tensor(bf_img[left:right, top:bottom])
                    if horizontal_flip:
                        bf_tile = torch.flip(bf_tile, [1])
                    if vertical_flip:
                        bf_tile = torch.flip(bf_tile, [0])
                    if rotation > 0:
                        bf_tile = torch.rot90(bf_tile, rotation, [0, 1])

                    try:
                        x[picture, i] = bf_tile
                    except RuntimeError as e:
                        print(f"Index error: {well}, {depth}")
                        print(f"Depth: {depth}")
                        print(f"Picture: {picture}")
                        print(f"Image: {i}")
                        print(f"Shape: {x.shape}")
                        print(f"Tile shape: {bf_tile.shape}")
                        print(f"Left: {left}, Right: {right}, Top: {top}, Bottom: {bottom}")

                        plt.imshow(mask)
                        plt.scatter(center[1], center[0], c='r', s=10, marker='x')
                        # draw bounding box
                        plt.plot([top, top], [right, left], 'b', linewidth=0.5)
                        plt.plot([top, bottom], [left, left], 'b', linewidth=0.5)
                        plt.plot([bottom, bottom], [left, right], 'b', linewidth=0.5)
                        plt.plot([bottom, top], [right, right], 'b', linewidth=0.5)
                        plt.savefig(f"error_{idx}.png")
                        raise e

                dead_tile = torch.tensor(dead_img[left:right, top:bottom])
                live_tile = torch.tensor(live_img[left:right, top:bottom])

                if horizontal_flip:
                    dead_tile = torch.flip(dead_tile, [1])
                    live_tile = torch.flip(live_tile, [1])
                if vertical_flip:
                    dead_tile = torch.flip(dead_tile, [0])
                    live_tile = torch.flip(live_tile, [0])
                if rotation > 0:
                    dead_tile = torch.rot90(dead_tile, rotation, [0, 1])
                    live_tile = torch.rot90(live_tile, rotation, [0, 1])

                y[picture, 0] = dead_tile
                y[picture, 1] = live_tile

                if self.debug and plt_pic < 4:
                    import matplotlib as mpl
                    mpl.rcParams['figure.dpi'] = 300
                    mpl.rcParams['font.size'] = 6

                    plt.subplot(2, 1, 1)
                    if plt_pic == 0:
                        plt.imshow(bf_img, cmap='gray', vmin=0, vmax=255)
                        plt.title('Brightfield')
                        plt.axis('off')
                    # Draw bounding box
                    plt.plot([top, top], [right, left], 'b', linewidth=0.5)
                    plt.plot([top, bottom], [left, left], 'b', linewidth=0.5)
                    plt.plot([bottom, bottom], [left, right], 'b', linewidth=0.5)
                    plt.plot([bottom, top], [right, right], 'b', linewidth=0.5)
                    plt.scatter(center[1], center[0], c='r', s=10, marker='x', linewidth=0.5)

                    plt.subplot(4, 2, 5)

                    if plt_pic == 0:
                        plt.imshow(mask, cmap='gray')
                        plt.title('Mask')
                        plt.axis('off')
                    plt.plot([top, top], [right, left], 'b', linewidth=0.5)
                    plt.plot([top, bottom], [left, left], 'b', linewidth=0.5)
                    plt.plot([bottom, bottom], [left, right], 'b', linewidth=0.5)
                    plt.plot([bottom, top], [right, right], 'b', linewidth=0.5)
                    plt.scatter(center[1], center[0], c='r', s=10, marker='x', linewidth=0.5)

                    plt_pic += 1

                    row = 0 if plt_pic < 3 else 1
                    col = plt_pic % 2 + 1

                    plt.subplot(8, 4, row * 4 + col + 2 + 16)
                    bf_tile = x[picture, 2]
                    plt.imshow(bf_tile, cmap='gray', vmin=0, vmax=255)
                    plt.title('Brightfield Tile ' + str(plt_pic))
                    plt.scatter(round(self.tile_size / 2), round(self.tile_size / 2), c='r', s=10, marker='x')
                    plt.axis('off')

                    plt.subplot(8, 4, 8 + row * 4 + col + 16)
                    plt.imshow(dead_tile, cmap='Greens', vmin=0, vmax=255)
                    plt.title('Dead Tile ' + str(plt_pic))
                    plt.scatter(round(self.tile_size / 2), round(self.tile_size / 2), c='r', s=10, marker='x')
                    plt.axis('off')

                    plt.subplot(8, 4, 10 + row * 4 + col + 16)
                    plt.imshow(live_tile, cmap='Oranges', vmin=0, vmax=255)
                    plt.title('Live Tile ' + str(plt_pic))
                    plt.scatter(round(self.tile_size / 2), round(self.tile_size / 2), c='r', s=10, marker='x')
                    plt.axis('off')

                    # plt.subplot(2, 3, 3)
                    # plt.imshow(self.masks[well], cmap='gray')
                    # plt.title('Mask')
                    # plt.scatter(center[1], center[0], c='r', s=10, marker='x')
                    # plt.axis('off')
                    #
                    # plt.subplot(2, 3, 6)
                    # plt.imshow(self.masks[well][left:right, top:bottom], cmap='gray')
                    # plt.title('Tile Mask')
                    # plt.scatter(round(self.tile_size / 2), round(self.tile_size / 2), c='r', s=10, marker='x')
                    # plt.axis('off')

            if self.transform is not None:
                if self.debug: print("Transforming...")
                x = self.transform(x)
                y = self.transform(y)

            if self.debug:
                plt.show()
                reply = input()
                if reply == "q":
                    self.debug = False

            # save potential centers to hard drive
            np.save(os.path.join(self.root, measurement, "potential_centers", well + ".npy"), potential_centers)

            if self.debug: print(f"Skipped tiles: {skipped_tiles}")

            return x, y

        except Exception as e:
            print(f"Error in well {well} and depth {depth}")
            print(f"With index {idx}")
            raise e

    def get_well_sample(self):
        well = np.random.choice(self.wells)
        measurement = well.split("f")[0][:-6]

        x = None
        active_tiles = None
        n_tiles = 1

        full_live = []
        full_dead = []
        full_bf = []

        for depth_idx, depth in enumerate(self.depths[well]):


            full_bf.append(cv2.imread(self.bf_stacks[well][depth], cv2.IMREAD_GRAYSCALE))
            full_dead.append(cv2.imread(self.dead_stacks[well][depth], cv2.IMREAD_GRAYSCALE))
            full_live.append(cv2.imread(self.live_stacks[well][depth], cv2.IMREAD_GRAYSCALE))


            x_i = torch.zeros((n_tiles, 5, self.tile_size, self.tile_size))
            for sub_depth_idx, sub_depth in enumerate(range(depth - self.depth_padding, depth + self.depth_padding + 1)):
                bf_img = cv2.imread(self.bf_stacks[well][sub_depth], cv2.IMREAD_GRAYSCALE)
                bf_img = equalize(bf_img, self.equalization_params_brightfield[measurement])

                tiles, _, _, active_tiles = self.tile_transform(bf_img, active_tiles)

                if n_tiles == 1:
                    print("n_tiles == 1")
                    print(f"Tiles: {len(tiles)}")
                    n_tiles = len(tiles)
                    x = torch.zeros((n_tiles * self.depth_range, 5, self.tile_size, self.tile_size))
                    x_i = torch.zeros((n_tiles, 5, self.tile_size, self.tile_size))

                for tile_idx, tile in enumerate(tiles):
                    x_i[tile_idx, sub_depth_idx] = torch.tensor(tile)

            x[depth_idx * n_tiles:(depth_idx + 1) * n_tiles] = x_i

        return active_tiles, x, n_tiles, (full_bf, full_dead, full_live)



def custom_collate_fn(batch):
    """
    Custom collate function to process batches of (x, y) pairs.

    Args:
        batch (list of tuples): A list where each element is a tuple (x, y),
                                with x having shape [4, 5, 64, 64] and
                                y having shape [4, 2, 64, 64].

    Returns:
        Tuple[Tensor, Tensor]: Two tensors, the first with shape [batch_size * 4, 5, 64, 64]
                               (inputs) and the second with shape [batch_size * 4, 2, 64, 64]
                               (outputs).
    """
    # Separate inputs (x) and outputs (y) in the batch
    xs, ys = zip(*batch)

    # Stack and reshape inputs (x)
    x_batch = torch.stack(xs, dim=0)
    batch_size, tiles_per_image, channels_x, height, width = x_batch.shape
    x_batch = x_batch.view(batch_size * tiles_per_image, channels_x, height, width)

    # Stack and reshape outputs (y)
    y_batch = torch.stack(ys, dim=0)
    # Assuming y has the same first two dimensions as x ([4, 2, 64, 64] for each sample)
    _, _, channels_y, _, _ = y_batch.shape
    y_batch = y_batch.view(batch_size * tiles_per_image, channels_y, height, width)

    return x_batch, y_batch
