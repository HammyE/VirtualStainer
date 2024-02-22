import os

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from MaximumIntensityProjection import MaximumIntensityProjection
from TileTransform import TileTransform


class HarmonyDataset(Dataset):
    def __init__(self, root, equalization=None, tile_size=128, overlap=32, transform=None):
        '''
        This is dataset class for harmony dataset, that parses the file structure and returns the brightfield images
        and the corresponding stained images.
        :param root: str, root directory of the dataset
        :param equalization: str, equalization method to be used (clahe, histogram, linear or None)
        :param transform: torchvision.Transforms, transform to be applied to the images
        '''
        self.main_dir = root
        self.transform = transform
        self.total_imgs = []
        self.labels = []
        self.equalization = equalization

        self.max_intensity = MaximumIntensityProjection(equalization_method=equalization)
        self.tile_transform = TileTransform(tile_size=tile_size, overlap=overlap)
        self.equalization_params_brightfield = {}
        self.equalization_params_dead = {}
        self.equalization_params_live = {}

        measurements = os.listdir(root)

        for index, measurement in enumerate(measurements):
            images = os.listdir(os.path.join(root, measurement, "images"))
            for image in images:
                if image.endswith("f01p01-ch1sk1fk1fl1.tiff"):
                    dead_channels = [os.path.join(root, measurement, "images", image)]
                    live_channels = [os.path.join(root, measurement, "images",
                                                  image.replace("-ch1sk1fk1fl1.tiff", "-ch2sk1fk1fl1.tiff"))]
                    bf_channels = [os.path.join(root, measurement, "images",
                                                image.replace("-ch1sk1fk1fl1.tiff", "-ch3sk1fk1fl1.tiff"))]
                    plane = 1
                    while True:
                        plane += 1
                        if plane < 10:
                            plane_str = "0" + str(plane)
                        else:
                            plane_str = str(plane)

                        # check if the file exists
                        if not os.path.isfile(os.path.join(root, measurement, "images",
                                                           image.replace("f01p01-ch1sk1fk1fl1.tiff",
                                                                         "f01p" + plane_str + "-ch1sk1fk1fl1.tiff"))):
                            break

                        dead_channels.append(
                            os.path.join(
                                root,
                                measurement,
                                "images",
                                image.replace(
                                    "f01p01-ch1sk1fk1fl1.tiff",
                                    "f01p" + plane_str + "-ch1sk1fk1fl1.tiff")
                            )
                        )
                        live_channels.append(
                            os.path.join(
                                root,
                                measurement,
                                "images",
                                image.replace(
                                    "f01p01-ch1sk1fk1fl1.tiff",
                                    "f01p" + plane_str + "-ch2sk1fk1fl1.tiff")
                            )
                        )
                        bf_channels.append(
                            os.path.join(
                                root,
                                measurement,
                                "images",
                                image.replace(
                                    "f01p01-ch1sk1fk1fl1.tiff",
                                    "f01p" + plane_str +
                                    "-ch3sk1fk1fl1.tiff")
                            )
                        )

                    self.total_imgs.append(bf_channels)
                    self.labels.append((dead_channels, live_channels))

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        brightfield = self.total_imgs[idx]

        # get the plate (parent folder
        plate = os.path.dirname(brightfield[0]).replace("dataset/", "").replace("/images", "")

        if plate not in self.equalization_params_brightfield.keys():
            # Get all images on the plate
            all_images_brightfield = []
            all_images_dead = []
            all_images_live = []
            for img in brightfield:
                if img.__contains__(plate):
                    all_images_brightfield.append(img)
                    all_images_dead.append(img.replace("ch3", "ch1"))
                    all_images_live.append(img.replace("ch3", "ch2"))

            # load images
            all_images_brightfield = [cv2.imread(img) for img in all_images_brightfield]
            all_images_dead = [cv2.imread(img) for img in all_images_dead]
            all_images_live = [cv2.imread(img) for img in all_images_live]
            self.equalization_params_brightfield[plate] = self.max_intensity.get_equalization_params(
                all_images_brightfield)
            self.equalization_params_dead[plate] = self.max_intensity.get_equalization_params(all_images_dead)
            self.equalization_params_live[plate] = self.max_intensity.get_equalization_params(all_images_live)

        # sample layer
        sample_idx = np.random.randint(0, len(brightfield))
        sample_img = cv2.imread(brightfield[sample_idx], 0)
        brightfield = self.max_intensity.equalize(sample_img, self.equalization_params_brightfield[plate])
        sample_dead = cv2.imread(self.labels[idx][0][sample_idx], 0)
        sample_live = cv2.imread(self.labels[idx][1][sample_idx], 0)
        dead = self.max_intensity.equalize(sample_dead, self.equalization_params_dead[plate])
        alive = self.max_intensity.equalize(sample_live, self.equalization_params_live[plate])

        # invert the image
        brightfield = cv2.bitwise_not(brightfield)

        # show the image

        # # Maximum intensity projection
        #
        # brightfield = [cv2.imread(img) for img in brightfield]
        #
        # # invert the image
        # brightfield = [cv2.bitwise_not(img) for img in brightfield]
        #
        #
        #
        # dead, alive = self.labels[idx]
        # dead = [cv2.imread(img) for img in dead]
        # alive = [cv2.imread(img) for img in alive]
        #
        # brightfield, dead, alive = self.max_intensity((brightfield, dead, alive))
        # brightfield, dead, alive = self.tile_transform((brightfield, dead, alive))
        #
        # # select random tile
        # print(len(brightfield))
        # idx = np.random.randint(0, len(brightfield))
        # brightfield = brightfield[idx]
        # dead = dead[idx]
        # alive = alive[idx]

        if self.transform is not None:
            brightfield = self.transform(brightfield)


        return brightfield, (dead, alive)
