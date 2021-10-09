import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from copy import deepcopy

root = 'store/datasets'


# Pretraining VAE model on CIFAR-10

def pretrain_vae(vae, device, bs=128, epochs=10):
    """ Function to pretrain given vae on CIFAR-10 """

    # Load data
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # CIFAR-10
    train_dataset = CIFAR10(root=root, download=True, train=True, transform=img_transform)
    test_dataset = CIFAR10(root=root, download=True, train=False, transform=img_transform)
    train_dl = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Train vae
    vae.to(device)
    print("\nPretraining VAE on CIFAR-10\n")
    for i in range(epochs):
        vae.train_epoch(train_dl, device)
        loss_value = vae.test_epoch(test_dl, device)
        print('Test set reconstruction error: %f' % loss_value)

    return vae


# Pretraining CNN on CIFAR-10

def pretrain_cnn(model, device, bs=128, epochs=10):
    """ Function to pretrain given model on CIFAR-10 """

    # Load data
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # CIFAR-10
    train_dataset = CIFAR10(root=root, download=True, train=True, transform=img_transform)
    test_dataset = CIFAR10(root=root, download=True, train=False, transform=img_transform)

    dl_train = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dl_test = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    if model.multihead:
        head_old = deepcopy(model.outputs[0])

    # Train model
    print("\nPretraining CNN on CIFAR-10\n")
    for i in range(epochs):
        print(f"Epoch {i + 1}\n-------------------------------")
        model.train_epoch(dl_train, loss_fn, device, task_nr=0, verbose=True)
        # Test model
        model.test_epoch(dl_test, loss_fn, device, task_nr=0, verbose=True)

    # Reset head used for pretrain
    if model.multihead:
        model.outputs[0] = head_old.to(device)

    return model
