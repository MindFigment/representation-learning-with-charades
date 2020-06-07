import torch
import torchvision
from image_folder_with_paths import ImageFolderWithPaths
from torch.nn import functional as F
from torch import nn, optim
import torchvision.transforms as T
import os
import numpy as np


class VAE(nn.Module):

    def __init__(self, latent_dim, nf):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        # Size of feature maps
        self.nf = nf

        print(self.nf, self.latent_dim)

        #####################
        ### BUILD ENCODER ###
        #####################

        self.encoder = nn.Sequential(
            # input: 3 x 224 x 224
            nn.Conv2d(3, nf, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # shape: nf x 224 x 224
            nn.Conv2d(nf, 2 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: nf x 112 x 112
            nn.Conv2d(2 * nf, 4 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: (2*nf) x 56 x 56
            nn.Conv2d(4 * nf, 8 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: (4*nf) x 28 x 28
            nn.Conv2d(8 * nf, 16 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: (8*ndf) x 14 x 14
            nn.Conv2d(16 * nf, 32 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: (16*ndf) x 7 x 7
            nn.Conv2d(32 * nf, 2 * self.latent_dim, 7, 1, 0),
            nn.LeakyReLU(0.2)
            # shape: (2*latent_dim) x 1 x 1
        )


        

        #####################
        ### BUILD DECODER ###
        #####################

        self.decoder = nn.Sequential(
            # input: 100 x 1 x 1
            nn.ConvTranspose2d(self.latent_dim, 32 * nf, 7, 1, 0),
            nn.LeakyReLU(0.2),
            # shape: 16 * nf x 7 x 7
            nn.ConvTranspose2d(32 * nf, 16 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: 8 * nf x 14 x 14
            nn.ConvTranspose2d(16 * nf, 8 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: 4 * nf x 28 x 28
            nn.ConvTranspose2d(8 * nf, 4 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: 2 * nf x 56 x 56
            nn.ConvTranspose2d(4 * nf, 2 * nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: nf x 112 x 112
            nn.ConvTranspose2d(2 * nf, nf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # shape: nf x 224 x 224
            nn.ConvTranspose2d(nf, 3, 3, 1, 1),
            nn.Sigmoid()
            # shape: 3 x 224 x 224
        )

    
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent code.

        Args:
            x: (Tensor) Input tensor to decoder [None, 3, 224, 224]

        Returns:
            (Tensor tuple) Tuple of tensors (mu, logvar)
        
        """

        encoded = self.encoder(x)
        # encoded = torch.squeeze(encoded)
        mu = encoded[:, :self.latent_dim, :, :]
        logvar = encoded[: ,self.latent_dim:, :, :]
        return mu, logvar


    def decode(self, latent_code):
        latent_code = latent_code.view(-1, self.latent_dim, 1, 1)
        return self.decoder(latent_code)

    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick

        Args:
            mu: (Tensor)  Mean matrix [None, latent_dim]
            logvar: (Tensor) Variance matrix [None, latent_dim]

        Returns:
            (Tensor) if train: sample from N(mu, std); if test: mu

        """

        if self.training:
            std = torch.exp(0.5 * logvar)
            # std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            # During inference, we simply split out the mean of the
            # learned distribution for the current input. We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability
            return mu

    
    def forward(self, x):
        mu, logvar = self.encode(x)
        latent_code = self.reparameterize(mu, logvar)
        # print('Latent code: {}, \n{}'.format(latent_code, latent_code.size()))
        return self.decode(latent_code), mu, logvar

    
    def get_feature_extractor(self):
        extractor = lambda x: self.encoder(x)[:, :self.latent_dim, :, :]
        return extractor

    
    def get_encoder(self):
        return self.encoder


    def loss_function(self, recon_x, x, mu, logvar):
        n = recon_x.size(0)
        bce = F.binary_cross_entropy(recon_x, x, reduction='sum') / n
        # bce = F.mse_loss(recon_x, x)
        # kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / n
        # bce tries to make our reconstruction as accurate as possible
        # kld tries to push the distributions as close as possible to unit Gaussian
        # print('BCE: {}'.format(bce))
        # print('KLD: {}'.format(kld))
        return bce + kld, bce, kld



if __name__ == '__main__':
    transform = torchvision.transforms.ToTensor()

    transform = T.Compose([T.Resize(size=256),
                           T.CenterCrop(size=224),
                           T.ToTensor()
    ])

    data_dir  = '/media/STORAGE/DATASETS/open-images/train/'
    dataset = ImageFolderWithPaths(data_dir, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128)
    iter_dataloader = iter(dataloader)
    batch = next(iter_dataloader)
    img = batch[0]

    model = VAE()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    result, mu, logvar = model(img)

    print(model.training)

    loss = model.loss_function(result, img, mu, logvar)

    print(len(list(model.parameters())))

    loss.backward()
    l = loss.item()
    optimizer.step()
    optimizer.zero_grad()
    print(l)
    print(loss.item())

    result, mu, logvar = model(img)
    loss = model.loss_function(result, img, mu, logvar)
    loss.backward()
    l = loss.item()
    optimizer.step()
    print(l)
    optimizer.zero_grad()

    result, mu, logvar = model(img)
    loss = model.loss_function(result, img, mu, logvar)
    loss.backward()
    l = loss.item()
    optimizer.step()
    optimizer.zero_grad()
    print(l)

    # print(result[0].size())
