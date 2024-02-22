import cv2
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# load in one bf, dead, and live image
bf_img = cv2.imread('dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c01f01p04-ch3sk1fk1fl1.tiff')
dead_img = cv2.imread('dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c01f01p04-ch1sk1fk1fl1.tiff')
live_img = cv2.imread('dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c01f01p04-ch2sk1fk1fl1.tiff')

# display the images with histograms
fig, axs = plt.subplots(2, 3, figsize=(30, 20))
axs[0, 0].imshow(bf_img)
axs[0, 0].set_title('Brightfield')
axs[0, 1].imshow(dead_img)
axs[0, 1].set_title('Dead')
axs[0, 2].imshow(live_img)
axs[0, 2].set_title('Live')

# Remove axis
axs[0, 0].axis('off')
axs[0, 1].axis('off')
axs[0, 2].axis('off')

# histograms
axs[1, 0].hist(np.array(bf_img).ravel(), bins=256, range=(0, 255), fc='k', ec='k')
axs[1, 0].set_title('Brightfield Histogram')
axs[1, 1].hist(np.array(dead_img).ravel(), bins=256, range=(0, 255), fc='g', ec='g')
axs[1, 1].set_title('Dead Histogram')
axs[1, 2].hist(np.array(live_img).ravel(), bins=256, range=(0, 255), fc='y', ec='y')
axs[1, 2].set_title('Live Histogram')
plt.show()

# Open a set of images and show a ridge graph of the histograms of the images
# load in one well bf, dead, and live images
bf_set = []
dead_set = []
live_set = []
labels = []

max_bf = 0
max_dead = 0
max_live = 0

for i in range(1, 2):
    formatted_i = str(i).zfill(2)

    bf_img = cv2.imread(
        f'dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c06f01p{formatted_i}-ch3sk1fk1fl1.tiff')
    dead_img = cv2.imread(
        f'dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c06f01p{formatted_i}-ch1sk1fk1fl1.tiff')
    live_img = cv2.imread(
        f'dataset/2307130102__2023-07-23T08_50_42-Measurement 11/Images/r01c06f01p{formatted_i}-ch2sk1fk1fl1.tiff')

    bf_image = np.array(bf_img).ravel()
    dead_image = np.array(dead_img).ravel()
    live_image = np.array(live_img).ravel()

    max_bf = max(max_bf, max(bf_image))
    max_dead = max(max_dead, max(dead_image))
    max_live = max(max_live, max(live_image))

    labels.append(f'p{formatted_i}')
    bf_set.append(bf_image)
    dead_set.append(dead_image)
    live_set.append(live_image)

print(max_bf, max_dead, max_live)

max_value = max(
    max([img.max() for img in bf_set]),
    max([img.max() for img in dead_set]),
    max([img.max() for img in live_set])
)


# Function to plot layered histograms for image sets
def plot_layered_histograms(image_sets, colors, labels, skip_layers=1):
    """
    Plot layered histograms for given image sets.

    Parameters:
    - image_sets: List of lists of images. Each sublist represents a set of images (e.g., bf_set).
    - colors: List of colors for each set.
    - labels: List of labels for each set.
    - skip_layers: Integer indicating how many layers to skip between plots.
    """
    # Determine the global maximum intensity value to standardize histogram bins
    max_intensity = max(max(img.max() for img in set) for set in image_sets)
    bins = np.linspace(0, max_intensity, 50)

    plt.figure(figsize=(10, 6))

    for image_set, color, label in zip(image_sets, colors, labels):
        for img in image_set[::skip_layers]:  # Skip layers as specified
            plt.hist(img.ravel(), bins=bins, color=color, alpha=0.3, label=label)

    # Simplify the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Deduplicate labels
    plt.legend(by_label.values(), by_label.keys())

    plt.title('Layered Histograms of Image Intensities')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()


for i in range(3):
    print(i)
    plot_layered_histograms(
        image_sets=[bf_set, dead_set, live_set][i],
        colors=['blue', 'green', 'orange'][i],
        labels=['BF', 'Dead', 'Live'][i],
        skip_layers=1  # Adjust as needed to reduce clutter
    )
