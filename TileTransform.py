import cv2
import numpy as np
import matplotlib.pyplot as plt


class TileTransform:
    '''
    This transform class is used to transform a brightfield image into a set of tiles of that image. If provided with
    stained images, the same transformation is applied to those images.
    '''
    def __init__(self, tile_size, overlap, transform=None):
        '''
        This is the constructor of the class.
        :param tile_size: int, size of the tiles
        :param overlap: int, overlap between the tiles
        :param transform: torchvision.transforms, transform to be applied to the images
        '''
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.debug = True

    def __call__(self, img_set):
        '''
        This method is called when the transform is applied to the images.
        :param img_set: tuple, set of images to be transformed
        :return: tuple, transformed images
        '''
        "Running TileTransform..."
        bf_img, dead_img, live_img = img_set

        bf_tiles, active_tiles = self.__tile_image(bf_img)

        if dead_img is not None:
            dead_tiles = self.__passive_tiling(dead_img, active_tiles)
        else:
            dead_tiles = None

        if live_img is not None:
            live_tiles = self.__passive_tiling(live_img, active_tiles)
        else:
            live_tiles = None

        return (bf_tiles, dead_tiles, live_tiles)

    def __tile_image(self, img):
        '''
        This method is used to tile the image.
        :param img: PIL.Image, image to be tiled
        :return: list, list of tiles
        '''
        img = np.array(img)

        if self.debug:
            print("Running active tiling...")
            # Show the image in max resolution
            fig = plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        # Create a mask of the big spheroid
        # Identify the biggest object in the image
        _, thresh = cv2.threshold(img, 47, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

        # Display the mask
        if self.debug:
            fig = plt.figure(figsize=(15, 15))
            plt.imshow(mask, cmap='gray')
            plt.axis('off')
            plt.show()
            input("Press Enter to continue...")

        mask_tiles = []
        tiles = []
        for i in range(0, img.shape[0], self.tile_size - self.overlap):
            for j in range(0, img.shape[1], self.tile_size - self.overlap):
                tile = img[i:i + self.tile_size, j:j + self.tile_size]
                mask_tile = mask[i:i + self.tile_size, j:j + self.tile_size]
                if tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size:
                    tiles.append(tile)
                    mask_tiles.append(mask_tile)




        # remove the tile if brightfield is all white
        tile_imgs = []
        active_tiles = []
        for idx, tile in enumerate(tiles):
            # if tile has
            if np.any(mask_tiles[idx]):
                tile_imgs.append(tile)
                active_tiles.append(idx)

        # show tiles if debug is True
        if self.debug:
            print("Showing tiles...")
            # Create a figure to display the images in position
            n_rows = int(np.ceil(np.sqrt(len(tiles))))
            n_cols = int(np.ceil(len(tiles) / n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            for i, tile in enumerate(tiles):
                ax = axs[i // n_cols, i % n_cols]
                if i in active_tiles:
                    ax.imshow(tile)
                ax.axis('off')
            plt.show()

        return tile_imgs, active_tiles

    def __passive_tiling(self,image, active_tiles):
        '''
        This method is used to tile the image.
        :param img: PIL.Image, image to be tiled
        :return: list, list of tiles
        '''

        if self.debug:
            print("Running passive tiling...")
            # Show the image in max resolution
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        img = np.array(image)
        tiles = []
        for i in range(0, img.shape[0], self.tile_size - self.overlap):
            for j in range(0, img.shape[1], self.tile_size - self.overlap):
                tile = img[i:i + self.tile_size, j:j + self.tile_size]
                if tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size:
                    tiles.append(tile)

        # show tiles if debug is True
        if self.debug:
            # Create a figure to display the images in position
            n_rows = int(np.ceil(np.sqrt(len(tiles))))
            n_cols = int(np.ceil(len(tiles) / n_rows))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
            for i, tile in enumerate(tiles):
                ax = axs[i // n_cols, i % n_cols]
                if i in active_tiles:
                    ax.imshow(tile)
                ax.axis('off')
            plt.show()

        # remove the tile if brightfield is all white
        tile_imgs = []
        for idx in active_tiles:
            tile_imgs.append(tiles[idx])
        return tile_imgs

def rolling_ball(image, kernel_size=15):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Perform the morphological opening operation
    background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Subtract the background from the original image
    corrected_image = cv2.subtract(image, background)

    # Optionally, you might want to add back a uniform background
    uniform_background = 0  # Adjust this value as needed
    corrected_image_with_uniform_background = cv2.add(corrected_image, uniform_background)

    return corrected_image_with_uniform_background



