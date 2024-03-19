'''This is the main file of the project. It will be used to run the project.'''

# Importing the necessary modules
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import HarmonyDataset, custom_collate_fn

# Defining the main function

data_dir = 'dataset'

if __name__ == '__main__':
    print("Running main.py")

    # load data
    transform = transforms.Compose([
        #transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(90)
    ])

    print("Loading dataset...")

    dataset = HarmonyDataset(
        root=data_dir,
        equalization="histogram",
        tile_size=128,
        overlap=16,
        transform=None,
        depth_padding=2,
        picture_batch_size=4
    )

    print("Dataset loaded.")

    print("Loading data loader...")
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    print("Data loader loaded.")

    # show data
    n_samples = 5
    print("Showing data...")

    images_processed = []
    for i, (images, labels) in enumerate(loader):

        if i >= n_samples:
            break
        print(images.shape)
        print(labels.shape)
        # calculate the memory usage of a batch
        print("Memory usage of input: ", images.element_size() * images.nelement() / 1024 / 1024, "MB")
        print("Memory usage of output: ", labels.element_size() * labels.nelement() / 1024 / 1024, "MB")
        print("Memory usage of both: ", images.element_size() * images.nelement() / 1024 / 1024 + labels.element_size() * labels.nelement() / 1024 / 1024, "MB")


        bf_img = images[0][2]
        dead_img = labels[0][0]
        live_img = labels[0][1]

        print(dead_img.shape)
        # from tensor to numpy, then to cv2
        bf_img = bf_img.numpy()
        dead_img = dead_img.numpy()
        live_img = live_img.numpy()

        images_processed.append((bf_img, dead_img, live_img))

    fig, axs = plt.subplots(n_samples, 3, figsize=(5, 10), dpi=300)
    for i, (bf_img, dead_img, live_img) in enumerate(images_processed):

        axs[i][0].imshow(bf_img, cmap='gray')
        axs[i][1].imshow(dead_img, cmap='Greens')
        axs[i][2].imshow(live_img, cmap='Oranges')

    axs[0][0].set_title('Brightfield')
    axs[0][1].set_title('Dead')
    axs[0][2].set_title('Live')

    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])

    plt.tight_layout()

    # Show the plot
    plt.show()


    # train model

    # test model

    # save model

    # extract materials


    pass