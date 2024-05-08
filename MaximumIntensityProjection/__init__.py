import multiprocessing

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev) ** 2)

def get_blur(img, mask):
    img = cv2.bitwise_and(img, img, mask=mask)
    return cv2.Laplacian(img, cv2.CV_64F).var()


def equalize(sample_img, param):
    '''
    A method to equalize the image.
    :param sample_img: np.array, image to be equalized
    :param param: tuple, equalization parameters
    :return: np.array, equalized image
    '''

    min_brightness, max_brightness = param
    sample_img = (sample_img - min_brightness) * (255.0 / (max_brightness - min_brightness))

    # Clip the values to the range [0, 255]
    sample_img = np.clip(sample_img, 0, 255).astype(np.uint8)

    return sample_img


def get_equalization_params(img_set, quantiles=None):
    """
    A method to get the equalization parameters for a plate.
    :return:
    """

    if quantiles is not None:
        brightness_count = {}
        for i in range(256):
            brightness_count[i] = 0

        print(f"Calculating brightness count for {len(img_set)}")
        print_idx = 0
        for image in img_set:
            if print_idx % 100 == 0:
                print(f"Processing image {print_idx}")
            image = cv2.imread(image)
            for i in range(256):
                brightness_count[i] += np.sum(image == i)

            print_idx += 1

        total_pixels = sum(brightness_count.values())
        min_brightness = 0
        max_brightness = 255
        min_count = 0
        max_count = total_pixels

        hist_min = 0
        hit_max = 255

        for i in range(256):
            min_count += brightness_count[i]
            if min_count > total_pixels * quantiles[0]:
                min_brightness = i
                break
            else:
                hist_min = i

        for i in range(255, -1, -1):
            max_count -= brightness_count[i]
            if max_count < total_pixels * quantiles[1]:
                max_brightness = i
                break
            if brightness_count[i] == 0:
                hist_max = i

        if min_brightness == max_brightness:
            max_brightness += 1

        ## For debugging show histogram and limits
        # keys = list(range(hist_min, hist_max + 1))
        # values = [brightness_count[i] for i in keys]
        # plt.bar(keys, values)
        # plt.axvline(min_brightness, color='r')
        # plt.axvline(max_brightness, color='r')
        # plt.show()

        return min_brightness, max_brightness


    min_brightness = float('inf')
    max_brightness = float('-inf')


    print(f"Calculating brightness count for {len(img_set)}")
    print_idx = 0
    for image in img_set:
        if print_idx % 100 == 0:
            print(f"Processing image {print_idx}")
        image = cv2.imread(image)
        min_brightness = min(min_brightness, np.min(image))
        max_brightness = max(max_brightness, np.max(image))
        print_idx += 1


    return min_brightness, max_brightness


class MaximumIntensityProjection(object):
    """MaximumIntensityProjection"""

    def __init__(self, equalization_method=None):
        '''
        Constructor of the class.
        :param regularization: str, equalization method to be used (clahe, histogram, linear or None)
        '''
        self.debug = False
        self.equalization_method = equalization_method  # "clahe", "histogram", "linear"

    def __call__(self, img_set, slices=None):
        bf_img_list, dead_img_list, live_img_list = img_set

        # Select a number of slices if specified, at a random depth
        if slices is not None:

            # Ensure the number of slices is not greater than the number of available slices
            if slices > len(bf_img_list):
                slices = len(bf_img_list)

            # Randomly select the sequence of slices
            max_start = len(bf_img_list) - slices
            start = np.random.randint(0, max_start)
            stop = start + slices

        else:
            start = None
            stop = None

        bf_channel = self.__one_channel_mip(bf_img_list, equalization=self.equalization_method, stop=stop, start=start,
                                            rolling_ball=False)
        dead_channel = self.__one_channel_mip(dead_img_list, equalization=self.equalization_method, stop=stop,
                                              start=start)
        live_channel = self.__one_channel_mip(live_img_list, equalization=self.equalization_method, stop=stop,
                                              start=start)

        if self.debug:
            fig, axs = plt.subplots(2, 3, figsize=(30, 20))
            axs[0, 0].imshow(bf_channel, cmap='gray')
            axs[0, 0].set_title('Brightfield MIP')
            axs[0, 1].imshow(dead_channel, cmap='Greens')
            axs[0, 1].set_title('Dead MIP')
            axs[0, 2].imshow(live_channel, cmap='Oranges')
            axs[0, 2].set_title('Live MIP')

            # Remove axis
            axs[0, 0].axis('off')
            axs[0, 1].axis('off')
            axs[0, 2].axis('off')

            # histograms
            axs[1, 0].hist(np.array(bf_channel).ravel(), bins=256, range=(0, 255), fc='k', ec='k')
            axs[1, 0].set_title('Brightfield Histogram')
            axs[1, 1].hist(np.array(dead_channel).ravel(), bins=256, range=(0, 255), fc='g', ec='g')
            axs[1, 1].set_title('Dead Histogram')
            axs[1, 2].hist(np.array(live_channel).ravel(), bins=256, range=(0, 255), fc='y', ec='y')
            axs[1, 2].set_title('Live Histogram')
            plt.show()

        return (bf_channel, dead_channel, live_channel)

    def __one_channel_mip(self, img_list, equalization=None, stop=None, start=None, rolling_ball=False):

        # Convert the list of PIL images to NumPy arrays
        arrays = [np.array(img) for img in img_list]

        if start is not None and stop is not None:
            arrays = arrays[start:stop]

        # perform rolling ball algorithm on every image
        if rolling_ball:
            for i in range(len(arrays)):
                arrays[i] = self.__rolling_ball(arrays[i])

        # Stack arrays to create a 3D array
        volume = np.stack(arrays, axis=0)

        # Perform Maximum Intensity Projection along the z-axis (axis=0)
        mip = np.amax(volume, axis=0)

        # Convert the result back to a PIL image
        mip_image = Image.fromarray(mip)

        # Apply equalization if specified
        if equalization == "clahe":
            mip_image = self.__clahe(mip)
        elif equalization == "histogram":
            mip_image = self.__histogram_equalization(mip)
        elif equalization == "linear":
            mip_image = self.__linear_contrast_stretching(mip)

        return mip_image

    def __repr__(self):
        return self.__class__.__name__ + '()'

    def __linear_contrast_stretching(self, image, min_percentile=1, max_percentile=99):
        """
        Apply linear contrast stretching to a single-channel image.

        Parameters:
        - image: Single-channel image as a NumPy array.
        - min_percentile: Lower percentile for contrast stretching.
        - max_percentile: Upper percentile for contrast stretching.

        Returns:
        - stretched_image: Image after applying linear contrast stretching.
        """
        # Ensure the image is in the correct format (8-bit or 16-bit single channel)

        # Check if the image is not single-channel and attempt to convert if it's in a known color format
        if len(image.shape) > 2 and image.shape[2] in [3, 4]:
            # Convert to grayscale if it's a 3-channel (RGB) or 4-channel (RGBA) image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Now ensure the image is either uint8 or uint16
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            # Assuming the image might be in another range, normalize and convert to uint8
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.dtype == np.uint16:
            # Convert to 8-bit by scaling if necessary
            # This conversion is crucial for compatibility with many OpenCV functions
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        # Calculate the percentile values
        min_val, max_val = np.percentile(image, (min_percentile, max_percentile))

        # Perform linear contrast stretching
        stretched_image = np.clip((image - min_val) * (255.0 / (max_val - min_val)), 0, 255).astype(np.uint8)

        return stretched_image

    def __histogram_equalization(self, image):
        """
        Apply histogram equalization to a single-channel image.

        Parameters:
        - image: Single-channel image as a NumPy array.

        Returns:
        - equalized_image: Image after applying histogram equalization.
        """
        # Ensure the image is in the correct format (8-bit or 16-bit single channel)

        # Check if the image is not single-channel and attempt to convert if it's in a known color format
        if len(image.shape) > 2 and image.shape[2] in [3, 4]:
            # Convert to grayscale if it's a 3-channel (RGB) or 4-channel (RGBA) image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Now ensure the image is either uint8 or uint16
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            # Assuming the image might be in another range, normalize and convert to uint8
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.dtype == np.uint16:
            # Convert to 8-bit by scaling if necessary
            # This conversion is crucial for compatibility with many OpenCV functions
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(image)

        return equalized_image

    def __clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        """
        Apply CLAHE to a single-channel image.

        Parameters:
        - image: Single-channel image as a NumPy array.
        - clip_limit: Threshold for contrast limiting.
        - tile_grid_size: Size of the grid for the histogram equalization.

        Returns:
        - clahe_image: Image after applying CLAHE.
        """
        # Ensure the image is in the correct format (8-bit or 16-bit single channel)

        # Check if the image is not single-channel and attempt to convert if it's in a known color format
        if len(image.shape) > 2 and image.shape[2] in [3, 4]:
            # Convert to grayscale if it's a 3-channel (RGB) or 4-channel (RGBA) image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Now ensure the image is either uint8 or uint16
        if image.dtype != np.uint8 and image.dtype != np.uint16:
            # Assuming the image might be in another range, normalize and convert to uint8
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if image.dtype == np.uint16:
            # Convert to 8-bit by scaling if necessary
            # This conversion is crucial for compatibility with many OpenCV functions
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        # Initialize CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Apply CLAHE
        clahe_image = clahe.apply(image)

        return clahe_image

    def __rolling_ball(self, image, kernel_size=15):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Perform the morphological opening operation
        background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # Subtract the background from the original image
        corrected_image = cv2.subtract(image, background)

        # Optionally, you might want to add back a uniform background
        uniform_background = 50  # Adjust this value as needed
        corrected_image_with_uniform_background = cv2.add(corrected_image, uniform_background)

        return corrected_image_with_uniform_background


def process_images(image_chunk, brightness_count, lock):
    local_count = {i: 0 for i in range(256)}
    print_idx = 0
    for image_path in image_chunk:
        if print_idx % 50 == 0:
            print(f"Processing image {print_idx}")
        print_idx += 1
        image = cv2.imread(image_path)
        for i in range(256):
            local_count[i] += np.sum(image == i)
    with lock:
        for i in range(256):
            brightness_count[i] += local_count[i]


def get_equalization_params_parallel(img_set, quantiles=None):
    num_workers = multiprocessing.cpu_count() * 2  # Number of available CPU cores
    chunk_size = len(img_set) // num_workers
    print(num_workers)

    # Set up a manager to handle shared data
    manager = multiprocessing.Manager()
    brightness_count = manager.dict({i: 0 for i in range(256)})
    lock = manager.Lock()

    # Create chunks of the image set
    image_chunks = [img_set[i:i + chunk_size] for i in range(0, len(img_set), chunk_size)]

    # Setup multiprocessing pool
    pool = multiprocessing.Pool(processes=num_workers)
    for chunk in image_chunks:
        pool.apply_async(process_images, args=(chunk, brightness_count, lock))
    pool.close()
    pool.join()

    total_pixels = sum(brightness_count.values())
    min_brightness = 0
    max_brightness = 255
    min_count = 0
    max_count = total_pixels

    # Calculate the histogram limits

    for i in range(256):
        min_count += brightness_count[i]
        if quantiles is not None:
            if min_count > total_pixels * quantiles[0]:
                min_brightness = i
                break


    for i in range(255, -1, -1):
        max_count -= brightness_count[i]
        if quantiles is not None:
            if max_count < total_pixels * quantiles[1]:
                max_brightness = i
                break

    if min_brightness == max_brightness:
        max_brightness += 1

    # Optionally, visualize the histogram and cutoffs
    # plt.bar(range(256), [brightness_count.get(i, 0) for i in range(256)])
    # plt.axvline(min_brightness, color='r')
    # plt.axvline(max_brightness, color='r')
    # plt.show()

    return min_brightness, max_brightness