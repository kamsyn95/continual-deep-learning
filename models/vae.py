import torch
import torch.nn as nn
import torch.nn.functional as F

from models.EncoderDecoder import Encoder, EncoderMLP, EncoderDCNN, Decoder, DecoderMLP, DecoderDCNN


class VariationalAutoencoder(nn.Module):
    """ Class for training/testing VAE and sampling from latent space """

    def __init__(self, net_type='MLP', size=28, in_channels=1, latent_dims=10, fc_size=256, beta=1.0, latent_classes=1):
        super(VariationalAutoencoder, self).__init__()

        if net_type == 'MLP':
            self.encoder = EncoderMLP(size, in_channels, fc_size, latent_dims, latent_classes)
            self.decoder = DecoderMLP(size, in_channels, fc_size, latent_dims, latent_classes)
        elif net_type == 'DeepCNN':
            self.encoder = EncoderDCNN(size, in_channels, latent_dims, fc_size, latent_classes)
            self.decoder = DecoderDCNN(size, in_channels, latent_dims, fc_size, latent_classes)
        else:
            self.encoder = Encoder(size, in_channels, 64, latent_dims, latent_classes)
            self.decoder = Decoder(size, in_channels, 64, latent_dims, latent_classes)

        self.latent_classes = latent_classes
        self.beta = beta    # kl_divergence coefficient
        self.optimizer = torch.optim.Adam(params=self.parameters())


    def forward(self, x, class_nr=0):
        latent_mu, latent_logvar = self.encoder(x, class_nr)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent, class_nr)
        return x_recon, latent_mu, latent_logvar


    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


    def vae_loss(self, recon_x, x, mu, logvar):
        # recon_x is the probability of a multivariate Bernoulli distribution p.
        # -log(p(x)) is then the pixel-wise binary cross-entropy.
        # Averaging or not averaging the binary cross-entropy over all pixels here
        # is a subtle detail with big effect on training, since it changes the weight
        # we need to pick for the other loss term by several orders of magnitude.
        # Not averaging is the direct implementation of the negative log likelihood,
        # but averaging makes the weight of the other loss term independent of the image resolution.

        recon_loss = F.binary_cross_entropy(recon_x.view(-1, self.decoder.output_size),
                                            x.view(-1, self.encoder.input_size), reduction='sum')

        # recon_loss = nn.BCELoss(reduction='sum')(recon_x, x) / x.size(0)

        # KL-divergence between the prior distribution over latent vectors
        # (the one we are going to sample from when generating new images)
        # and the distribution estimated by the generator for the given image.
        kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + (self.beta * kldivergence)


    def train_batch(self, image_batch, device, class_nr=0):
        image_batch = image_batch.to(device)
        # vae reconstruction
        image_batch_recon, latent_mu, latent_logvar = self.forward(image_batch, class_nr)
        # reconstruction error
        loss = self.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # one step of the optmizer (using the gradients from backpropagation)
        self.optimizer.step()
        return loss


    def train_epoch(self, dataloader, device, class_nr=0):
        self.train()
        num_batches, train_loss = 0, 0.

        for image_batch, _ in dataloader:
            loss = self.train_batch(image_batch, device, class_nr)
            train_loss += loss.item()
            num_batches += 1

        return train_loss / num_batches


    def test_epoch(self, dataloader, device, class_nr=0):
        self.eval()
        num_batches = 0
        test_loss = 0

        for image_batch, _ in dataloader:
            with torch.no_grad():
                image_batch = image_batch.to(device)
                # vae reconstruction
                image_batch_recon, latent_mu, latent_logvar = self.forward(image_batch, class_nr)
                # reconstruction error
                loss = self.vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
                test_loss += loss.item()
                num_batches += 1

        test_loss /= num_batches
        return test_loss


    def sample(self, no_of_samples, device, class_nr=0):
        self.eval()
        self.decoder.eval()
        with torch.no_grad():
            # sample latent vectors from the normal distribution
            latent = torch.randn(no_of_samples, self.decoder.latent_dims, device=device)
            # reconstruct images from the latent vectors
            img_recon = self.decoder(latent, class_nr)
        return img_recon
