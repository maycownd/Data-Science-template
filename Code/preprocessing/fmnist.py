import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import torch

PATH_TRAIN = "../../Data/raw/fashion-mnist_train.csv"
PATH_TEST = "../../Data/raw/fashion-mnist_test.csv"


class DatasetFashionMNIST(Dataset):
    """Fashion MNIST dataset using Pytorch class Dataset.

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, path, transform=None):
        """Method to initilaize variables."""
        self.fashion_MNIST = pd.read_csv(path).values
        self.transform = transform

        # first column is of labels.
        self.labels = self.fashion_MNIST[:, 0]

        # Dimension of Images = 28 * 28 * 1.
        # where height = width = 28 and color_channels = 1.
        self.images = self.fashion_MNIST[:, 1:]\
            .reshape(-1, 28, 28, 1).astype('float32')

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.images[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)


def load_fmnist_torch():
    """Load Train and Test DatasetFashionMNIST
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
        ]
    )
    trainset = DatasetFashionMNIST(path=PATH_TRAIN, transform=transform)
    testset = DatasetFashionMNIST(path=PATH_TEST, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, shuffle=True, batch_size=64)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False, batch_size=64)
    return trainloader, testloader


if __name__ == "__main__":
    load_fmnist_torch()
