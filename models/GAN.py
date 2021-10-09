# ------------------------------------------------------------------------------
#    Generative Adversarial Networks
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, nc=3, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class GAN(nn.Module):
    def __init__(self, nc=3, nz=100, ndf=64, ngf=64, lr=0.0002, betas=(0.9, 0.999)):
        super().__init__()
        # Generator Network
        self.netG = Generator(nc, nz, ngf)
        self.netG.apply(self.weights_init)

        # Discriminator Network
        self.netD = Discriminator(nc, ndf)
        self.netD.apply(self.weights_init)

        self.criterion = nn.BCELoss()
        self.optimizerD = optim.Adam(self.netD.parameters(), lr, betas)
        self.optimizerG = optim.Adam(self.netG.parameters(), lr, betas)

        self.nz = nz
        self.real_label = 1
        self.fake_label = 0


    def forward(self, input):
        outG = self.netG(input)
        output = self.netD(outG)
        return output


    def weights_init(self, m):
        """ custom weights initialization called on netG and netD """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


    def train_batch(self, data, device):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.netD.zero_grad()
        # real_cpu = data[0].to(device)
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), self.real_label, device=device, dtype=torch.float32)
        # print('data', real_cpu.shape)

        output = self.netD(real_cpu)
        errD_real = self.criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, self.nz, 1, 1, device=device)
        fake = self.netG(noise)
        label.fill_(self.fake_label)
        output = self.netD(fake.detach())
        errD_fake = self.criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.netG.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        output = self.netD(fake)
        errG = self.criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

        return D_x, errD, errG, D_G_z1, D_G_z2


    def train_epoch(self, dataloader, device, epoch, niter, batch_size=64, outf=''):
        fixed_noise = torch.randn(batch_size, self.nz, 1, 1, device=device)

        for i, data in enumerate(dataloader, 0):
            D_x, errD, errG, D_G_z1, D_G_z2 = self.train_batch(self, data, device)

            if i % 50 == 0:
                print('[%d/%d][%d/%d]' % (epoch, niter, i, len(dataloader)))

        print('Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if outf != '':
            fake = self.netG(fixed_noise)
            vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (outf, epoch), normalize=True)
            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))


    def sample(self, no_of_samples, device):
        self.eval()
        self.netG.eval()
        with torch.no_grad():
            # sample latent vectors from the normal distribution
            latent = torch.randn(no_of_samples, self.nz, 1, 1, device=device)
            # reconstruct images from the latent vectors
            img_recon = self.netG(latent)
        return img_recon
