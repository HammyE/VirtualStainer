import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _display_tile(tiles, active_tiles, cmap):
    return
    # Create a figure to display the images in position
    n_rows = int(np.ceil(np.sqrt(len(tiles))))
    n_cols = int(np.ceil(len(tiles) / n_rows))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    for i, tile in enumerate(tiles):
        ax = axs[i // n_cols, i % n_cols]
        if i in active_tiles:
            ax.imshow(tile, cmap=cmap)
            rect = patches.Rectangle((0, 0), 1, 1, linewidth=5, edgecolor='g', facecolor='none',
                                     transform=ax.transAxes)
        else:
            ax.imshow(tile, cmap=cmap, alpha=0.3)
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color='gray', alpha=0.5)
        ax.add_patch(rect)
        ax.axis('off')
    plt.show()

def _display_image(image, cmap):
    # Show the image in max resolution
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()


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
        self.debug = False

    def __call__(self, img_set, active_tiles=None):
        '''
        This method is called when the transform is applied to the images.
        :param img_set: tuple, set of images to be transformed
        :return: tuple, transformed images
        '''
        "Running TileTransform..."
        bf_img, dead_img, live_img = img_set

        if active_tiles is None:
            bf_tiles, active_tiles = self.__tile_image(bf_img)
        else:
            bf_tiles = self.__passive_tiling(bf_img, active_tiles)

        if dead_img is not None:
            dead_tiles = self.__passive_tiling(dead_img, active_tiles, cmap="Greens")
        else:
            dead_tiles = None

        if live_img is not None:
            live_tiles = self.__passive_tiling(live_img, active_tiles, cmap="Oranges")
        else:
            live_tiles = None

        return (bf_tiles, dead_tiles, live_tiles, active_tiles)

    def get_mask(self, img):
        """
        This method is used to get the mask of the spheroid.
        :param img:
        :return:
        """

        # Create a mask of the big spheroid
        # Identify the biggest object in the image

        img = np.array(img)
        img = 256 - img

        img = cv2.convertScaleAbs(img)
        _, thresh = cv2.threshold(img, 47, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)

        if self.debug:
            _display_image(mask, cmap="gray")
            input()

        return mask

    def __tile_image(self, img, cmap="gray"):
        """
        This method is used to tile the image.
        :param img: PIL.Image, image to be tiled
        :return: list, list of tiles
        """

        mask = self.get_mask(img)

        # Display the mask
        if self.debug:
            _display_image(img, cmap=cmap)

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
            if np.quantile(mask_tiles[idx], q=0.8) > 0:
                tile_imgs.append(tile)
                active_tiles.append(idx)

        # show tiles if debug is True
        if self.debug:
            _display_tile(tiles, active_tiles, cmap)
            _display_tile(mask_tiles, active_tiles, cmap)


        return tile_imgs, active_tiles

    def __passive_tiling(self,image, active_tiles, cmap="gray"):
        '''
        This method is used to tile the image.
        :param img: PIL.Image, image to be tiled
        :return: list, list of tiles
        '''

        if self.debug:
            _display_image(image, cmap=cmap)

        img = np.array(image)
        tiles = []
        for i in range(0, img.shape[0], self.tile_size - self.overlap):
            for j in range(0, img.shape[1], self.tile_size - self.overlap):
                tile = img[i:i + self.tile_size, j:j + self.tile_size]
                if tile.shape[0] == self.tile_size and tile.shape[1] == self.tile_size:
                    tiles.append(tile)

        # show tiles if debug is True
        if self.debug:
            _display_tile(tiles, active_tiles, cmap)

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



