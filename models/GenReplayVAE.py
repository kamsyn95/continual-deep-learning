# ------------------------------------------------------------------------------
#    Generative Replay with Variational Autoencoder - VAE
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
import torchvision.utils as vutils
from data.utils import CustomImageDataset

from models.vae import VariationalAutoencoder
from models.base_nets import MLP, CNN, DeepCNN


class GenerativeReplay(nn.Module):
    """Class to perform generative replay of old data samples using VAE"""

    def __init__(self, net_type='MLP', vae_type='MLP', in_size=28, in_channels=1, vae_latent_dims=10, c=16, **kwargs):
        super().__init__()

        # Different latent space for each task in class incremental learning
        lc = kwargs['tasks'] if kwargs['scenario'] == 'class' else 1

        if net_type == 'CNN':
            self.model = CNN(in_size, in_channels, c, **kwargs)
        elif net_type == 'DeepCNN':
            self.model = DeepCNN(in_size, in_channels, c, **kwargs)
        else:
            self.model = MLP(in_size, in_channels, **kwargs)

        self.vae = VariationalAutoencoder(net_type=vae_type, size=in_size, in_channels=in_channels, latent_classes=lc,
                                          latent_dims=vae_latent_dims, fc_size=kwargs['fc_size'], beta=1.0)

        self.save_dir = 'store/imgs/VAE/'


    def forward(self, x):
        self.model(x)
        return x


    def generate_dataset(self, size, device, task_nr, batch_size=64):
        """ Generate dataset - sampling from generator and labelling by model """

        self.model.task_nr = task_nr
        class_nr = task_nr if self.model.scenario == 'class' else 0
        active_classes = self.model.classes_per_task * (task_nr + 1)
        num_batches = int(np.floor(size / batch_size))

        for b in range(num_batches):
            # Sample Latent Vector from Prior (VAE as Generator)
            img_recon = self.vae.sample(batch_size, device, class_nr)

            # Create labels
            logits = self.model.forward(img_recon)
            if self.model.scenario == 'class':
                logits = logits[:, :active_classes]
            y_recon = logits.argmax(1)

            # Create dataset
            if len(img_recon.shape) == 3:
                img_recon = torch.reshape(img_recon, (batch_size, 1, img_recon.shape[-2], img_recon.shape[-1]))
            if b == 0:
                dataset = CustomImageDataset(img_recon, y_recon)
            else:
                # Concat current data to dataset
                dataset.imgs = torch.cat((dataset.imgs, img_recon), dim=0)
                dataset.img_labels = torch.cat((dataset.img_labels, y_recon))

        dataset_loader = DataLoader(dataset, batch_size=batch_size)
        return dataset_loader


    def train_epoch(self, MultipleDLs, loss_fn, device, task_nr, epoch, verbose=True, save=True):
        self.model.train()
        self.vae.train()
        class_nr = task_nr if self.model.scenario == 'class' else 0

        for batch_nr, batch in enumerate(MultipleDLs):
            loss_value, vae_loss_value = 0., 0.
            losses = []

            for i, dl in enumerate(batch):
                # Unpack values
                X, y = dl[0], dl[1]
                # -------------TRAIN TASK ------------#
                self.model.task_nr = i
                # Train on batch
                loss = self.model.train_batch_v2(X, y, loss_fn, device)
                loss_value += loss.item()
                losses.append(loss)

                # Train generator
                vae_loss = self.vae.train_batch(image_batch=X, device=device, class_nr=class_nr)
                vae_loss_value += vae_loss.item()

            # Update parameters using combined loss
            self.model.optimizer.zero_grad()
            total_loss = (1 / (task_nr+1)) * losses.pop(-1)   # current loss
            for prev_loss in losses:
                total_loss += (1 - (1 / (task_nr+1))) * prev_loss   # previous losses
            total_loss.backward(retain_graph=False)
            self.model.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose and (batch_nr % 50 == 0):
                current = batch_nr * 256
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        if verbose:
            print('Average reconstruction error: %f' % (vae_loss_value / batch_nr))

        if save:
            img_recon = self.vae.sample(100, device, class_nr=class_nr)
            if len(img_recon.shape) == 3:
                img_recon = torch.reshape(img_recon, (100, 1, img_recon.shape[-2], img_recon.shape[-1]))

            vutils.save_image(img_recon.detach(), self.save_dir + 'VAE_fake_samples_task_%02d_epoch_%03d.png'
                              % (task_nr, epoch), normalize=True)

        return loss_value, vae_loss_value


    def train_epoch_model(self, MultipleDLs, loss_fn, device, task_nr, verbose=True):
        self.model.train()
        for batch_nr, batch in enumerate(MultipleDLs):
            loss_value = 0.0
            losses = []
            for i, dl in enumerate(batch):
                # Unpack values
                X, y = dl[0], dl[1]
                # -------------TRAIN TASK ------------#
                self.model.task_nr = i
                # Train on batch
                loss = self.model.train_batch_v2(X, y, loss_fn, device)
                loss_value += loss.item()
                losses.append(loss)

            # Update parameters using combined loss
            self.model.optimizer.zero_grad()
            total_loss = (1 / (task_nr+1)) * losses.pop(-1)   # current loss
            for prev_loss in losses:
                total_loss += (1 - (1 / (task_nr+1))) * prev_loss   # previous losses
            total_loss.backward(retain_graph=False)
            self.model.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose:
                if batch_nr % 50 == 0:
                    current = batch_nr * 256
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_value


    def train_epoch_generator(self, MultipleDLs, device, task_nr, epoch, verbose=True, save=True):
        self.vae.train()
        class_nr = task_nr if self.model.scenario == 'class' else 0

        for batch_nr, batch in enumerate(MultipleDLs):
            vae_loss_value = 0.0
            for i, dl in enumerate(batch):
                # Unpack values
                X, y = dl[0], dl[1]
                # Train generator
                vae_loss = self.vae.train_batch(image_batch=X, device=device, class_nr=class_nr)
                vae_loss_value += vae_loss.item()

        if verbose:
            print('Epoch %d: Average reconstruction error: %f' % (epoch, (vae_loss_value / batch_nr)))

        if save:
            img_recon = self.vae.sample(100, device, class_nr=class_nr)
            if len(img_recon.shape) == 3:
                img_recon = torch.reshape(img_recon, (100, 1, img_recon.shape[-2], img_recon.shape[-1]))

            vutils.save_image(img_recon.detach(), self.save_dir + 'VAE_fake_samples_task_%02d_epoch_%03d.png'
                              % (task_nr, epoch), normalize=True)

        return vae_loss_value
