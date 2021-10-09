import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderMLP(nn.Module):
    def __init__(self, in_size=28, in_channels=1, fc_size=400, latent_dims=10, latent_classes=1):
        super(EncoderMLP, self).__init__()
        self.input_size = in_size
        self.in_channels = in_channels
        self.latent_dims = latent_dims
        self.hidden_size = fc_size

        # Network architecture
        self.fc1 = nn.Linear(in_size * in_size * in_channels, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        # different latent space for each tasks
        self.fc_mu = nn.ModuleList([nn.Linear(self.hidden_size, self.latent_dims) for _ in range(latent_classes)])
        self.fc_logvar = nn.ModuleList([nn.Linear(self.hidden_size, self.latent_dims) for _ in range(latent_classes)])

    def forward(self, x, class_nr=0):
        x = x.view(-1, self.input_size * self.input_size * self.in_channels)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_mu = self.fc_mu[class_nr](x)
        x_logvar = self.fc_logvar[class_nr](x)
        return x_mu, x_logvar


class DecoderMLP(nn.Module):
    def __init__(self, out_size=28, out_channels=1, fc_size=400, latent_dims=10, latent_classes=1):
        super(DecoderMLP, self).__init__()
        self.output_size = out_size
        self.out_channels = out_channels
        self.latent_dims = latent_dims
        self.hidden_size = fc_size

        # Network architecture
        # different latent space for each tasks
        self.fc_in = nn.ModuleList([nn.Linear(self.latent_dims, self.hidden_size) for _ in range(latent_classes)])
        self.fc2 = nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size)
        self.fc1 = nn.Linear(in_features=self.hidden_size, out_features=out_size * out_size * out_channels)

    def forward(self, x, class_nr=0):
        x = F.relu(self.fc_in[class_nr](x))
        x = F.relu(self.fc2(x))
        # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = self.fc1(x)
        x = x.view(-1, self.out_channels, self.output_size, self.output_size)
        x = torch.sigmoid(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_size=28, in_channels=1, c=64, latent_dims=10, latent_classes=1):
        super(Encoder, self).__init__()
        self.input_size = in_size
        self.latent_dims = latent_dims
        self.c = c
        # Network architecture
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=4, stride=2,
                               padding=1)  # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)  # out: c x 7 x 7

        # different latent space for each tasks
        img_dim = 7 if in_size == 28 else 8
        self.fc_mu = nn.ModuleList([nn.Linear(c * 2 * img_dim * img_dim, self.latent_dims)
                                    for _ in range(latent_classes)])
        self.fc_logvar = nn.ModuleList([nn.Linear(c * 2 * img_dim * img_dim, self.latent_dims)
                                        for _ in range(latent_classes)])

    def forward(self, x, class_nr=0):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu[class_nr](x)
        x_logvar = self.fc_logvar[class_nr](x)
        return x_mu, x_logvar


class Decoder(nn.Module):
    def __init__(self, out_size=28, out_channels=1, c=64, latent_dims=10, latent_classes=1):
        super(Decoder, self).__init__()
        self.output_size = out_size
        self.latent_dims = latent_dims
        self.c = c

        # Network architecture
        # different latent space for each tasks
        img_dim = 7 if out_size == 28 else 8
        self.fc_in = nn.ModuleList([nn.Linear(self.latent_dims, c * 2 * img_dim * img_dim)
                                    for _ in range(latent_classes)])
        self.conv2 = nn.ConvTranspose2d(in_channels=c * 2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, class_nr=0):
        x = self.fc_in[class_nr](x)
        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        img_dim = 7 if self.output_size == 28 else 8
        x = x.view(x.size(0), self.c * 2, img_dim, img_dim)
        x = F.relu(self.conv2(x))
        # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = torch.sigmoid(self.conv1(x))
        return x


class EncoderDCNN(nn.Module):

    def __init__(self, in_size=32, in_channels=3, latent_dims=100, fc_size=256, latent_classes=1):
        super(EncoderDCNN, self).__init__()

        self.input_size = in_size
        self.latent_dims = latent_dims
        self.in_channels = in_channels
        self.fc_size = fc_size
        self.capacity = 16
        c = self.capacity

        # Network architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.in_channels, c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),

            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),

            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),

            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(),

            nn.Conv2d(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 16),
            nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Linear((c * 16) * 2 * 2, self.fc_size),  # 2*2 from image dimension,
            nn.ReLU(),
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU()
        )
        # different latent space for each tasks
        self.fc_mu = nn.ModuleList([nn.Linear(self.fc_size, self.latent_dims) for _ in range(latent_classes)])
        self.fc_logvar = nn.ModuleList([nn.Linear(self.fc_size, self.latent_dims) for _ in range(latent_classes)])

    def forward(self, x, class_nr=0):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.linear_layers(x)
        x_mu = self.fc_mu[class_nr](x)
        x_logvar = self.fc_logvar[class_nr](x)
        return x_mu, x_logvar


class DecoderDCNN(nn.Module):

    def __init__(self, out_size=32, out_channels=3, latent_dims=100, fc_size=256, latent_classes=1):
        super(DecoderDCNN, self).__init__()

        self.output_size = out_size
        self.latent_dims = latent_dims
        self.out_channels = out_channels
        self.fc_size = fc_size
        self.capacity = 16
        c = self.capacity

        # Network architecture
        # different latent space for each tasks
        self.fc_in = nn.ModuleList([nn.Linear(self.latent_dims, self.fc_size) for _ in range(latent_classes)])
        self.linear_layers = nn.Sequential(
            nn.Linear(self.fc_size, self.fc_size),
            nn.ReLU(),
            nn.Linear(self.fc_size, (c * 16) * 2 * 2),
            nn.ReLU()
        )
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(c * 2, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),

            nn.ConvTranspose2d(c, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x, class_nr=0):
        x = self.fc_in[class_nr](x)
        x = F.relu(x)
        x = self.linear_layers(x)
        # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = x.view(x.size(0), self.capacity * 16, 2, 2)
        x = self.deconv_layers(x)
        # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        x = torch.sigmoid(x)
        return x
