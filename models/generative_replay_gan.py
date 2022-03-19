# ------------------------------------------------------------------------------
#    Generative Replay with Generative Adversarial Networks - GAN
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from data.utils import CustomImageDataset
import torchvision.utils as vutils
import numpy as np

from models.gan_modules import GAN
from models.base_nets import MLP, CNN, DeepCNN


class GenerativeReplay(nn.Module):
    """Class to perform generative replay of old data samples using GAN"""

    def __init__(
        self,
        net_type="MLP",
        lr_gan=0.0002,
        betas_gan=(0.5, 0.999),
        latent_dims=100,
        c=16,
        fc_size=400,
        batch_size=64,
        **kwargs,
    ):

        super().__init__()

        if net_type == "CNN":
            self.model = CNN(**kwargs)
        elif net_type == "DeepCNN":
            self.model = DeepCNN(c=c, fc_size=fc_size, **kwargs)
        else:
            self.model = MLP(**kwargs)

        self.gan = GAN(
            nc=kwargs["in_channels"], nz=latent_dims, ndf=batch_size, ngf=batch_size, lr=lr_gan, betas=betas_gan
        )

        self.save_dir = "store/imgs/GAN/"

    def forward(self, x):
        self.model(x)
        return x

    def generate_dataset(self, size, device, task_nr, batch_size=64):
        """Generate dataset - sampling from generator and labelling by model"""

        self.model.task_nr = task_nr
        active_classes = self.model.classes_per_task * (task_nr + 1)
        num_batches = int(np.floor(size / batch_size))

        for b in range(num_batches):
            # Sample Latent Vector from Prior (GAN as Generator)
            img_recon = self.gan.sample(batch_size, device)

            # Create labels
            logits = self.model.forward(img_recon)
            if self.model.scenario == "class":
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
        self.gan.train()

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

                # Train generator
                D_x, errD, errG, D_G_z1, D_G_z2 = self.gan.train_batch(data=X, device=device)

            # Update parameters using combined loss
            self.model.optimizer.zero_grad()
            total_loss = (1 / (task_nr + 1)) * losses.pop(-1)  # current loss
            for prev_loss in losses:
                total_loss += (1 - (1 / (task_nr + 1))) * prev_loss  # previous losses
            total_loss.backward(retain_graph=False)
            self.model.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose and (batch_nr % 50 == 0):
                current = batch_nr * 256
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        if verbose:
            print(
                "Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
            )

        if save:
            fixed_noise = torch.randn(64, self.gan.nz, 1, 1, device=device)
            fake = self.gan.netG(fixed_noise)
            vutils.save_image(
                fake.detach(),
                self.save_dir + "GAN_fake_samples_task_%02d_epoch_%03d.png" % (task_nr, epoch),
                normalize=True,
            )

        return loss_value

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
            total_loss = (1 / (task_nr + 1)) * losses.pop(-1)  # current loss
            for prev_loss in losses:
                total_loss += (1 - (1 / (task_nr + 1))) * prev_loss  # previous losses
            total_loss.backward(retain_graph=False)
            self.model.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose and (batch_nr % 50 == 0):
                current = batch_nr * 256
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_value

    def train_epoch_generator(self, MultipleDLs, device, task_nr, epoch, verbose=True, save=True):
        self.gan.train()
        for batch_nr, batch in enumerate(MultipleDLs):
            for i, dl in enumerate(batch):
                # Unpack values
                X, _ = dl[0], dl[1]
                # Train generator
                D_x, errD, errG, D_G_z1, D_G_z2 = self.gan.train_batch(data=X, device=device)

        if verbose:
            print(
                "Epoch %d:  Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f"
                % (epoch, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
            )

        if save:
            fixed_noise = torch.randn(64, self.gan.nz, 1, 1, device=device)
            fake = self.gan.netG(fixed_noise)
            vutils.save_image(
                fake.detach(),
                self.save_dir + "GAN_fake_samples_task_%02d_epoch_%03d.png" % (task_nr, epoch),
                normalize=True,
            )
