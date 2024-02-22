'''This is the main file of the project. It will be used to run the project.'''

# Importing the necessary modules
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import HarmonyDataset

# Defining the main function

data_dir = 'dataset'

if __name__ == '__main__':

    # load data
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = HarmonyDataset(
        root=data_dir,
        equalization="histogram",
        tile_size=256,
        overlap=32,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
    )

    # show data
    n_samples = 5

    fig, axs = plt.subplots(n_samples, 3, figsize=(5, 10), dpi=300)
    for i, (images, labels) in enumerate(loader):
        print(images.shape)
        print(labels[0].shape)
        print(labels[1].shape)

        if i >= n_samples:
            break

        bf_img = images[0]
        #print(labels[i][0].shape)
        dead_img = labels[0][0]
        live_img = labels[1][0]
        # from tensor to PIL
        bf_img = transforms.ToPILImage()(bf_img)
        dead_img = transforms.ToPILImage()(dead_img)
        live_img = transforms.ToPILImage()(live_img)

        axs[i][0].imshow(bf_img)
        axs[i][1].imshow(dead_img)
        axs[i][2].imshow(live_img)

    axs[0][0].set_title('Brightfield')
    axs[0][1].set_title('Dead')
    axs[0][2].set_title('Live')


    #remove axis
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])

    plt.tight_layout()


    plt.show()
        # show sample


    # preprocess data

    # train model

    # test model

    # save model

    # extract materials


    pass