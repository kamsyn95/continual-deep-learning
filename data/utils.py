from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms


class CustomImageDataset(Dataset):
    def __init__(self, imgs, labels, transform=None, target_transform=None):
        self.img_labels = labels
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.imgs[idx, :, :, :]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class CatDataloaders:
    """Class to concatenate multiple dataloaders"""

    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        len(self.dataloaders)

    def __iter__(self):
        self.loader_iter = []
        for data_loader in self.dataloaders:
            self.loader_iter.append(iter(data_loader))
        return self

    def __next__(self):
        out = []
        for data_iter in self.loader_iter:
            out.append(next(data_iter))  # may raise StopIteration
        return tuple(out)


def load_mnist_fmnist(root='store/datasets'):
    """Function to load MNIST and Fashion MNIST datasets"""

    # Images padded to size 32 x 32
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(2)
    ])
    # Load MNIST
    mnist_train = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    # Load Fashion MNIST
    f_mnist_train = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    f_mnist_test = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    # return lists of train and test datasets and some additional info
    train_datasets = [mnist_train, f_mnist_train]
    test_datasets = [mnist_test, f_mnist_test]
    config = {'size': 32, 'channels': 1, 'classes': 10}
    classes_per_task = 10

    return (train_datasets, test_datasets), config, classes_per_task
