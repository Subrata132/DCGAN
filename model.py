import torch
import torch.nn as nn


def single_block_disc(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      flag=None):
    if flag:
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.LeakyReLU(0.2),
        )
    else:
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    return block


def single_block_gen(in_channels,
                     out_channels,
                     kernel_size,
                     stride,
                     padding,
                     flag=None):
    if flag:
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ),
            nn.Tanh()
        )
    else:
        block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return block


class Discriminator(nn.Module):
    def __init__(self, no_channels, feature_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            single_block_disc(no_channels, feature_dim, 4, 2, 1, True),
            single_block_disc(feature_dim, feature_dim * 2, 4, 2, 1),
            single_block_disc(feature_dim * 2, feature_dim * 4, 4, 2, 1),
            single_block_disc(feature_dim * 4, feature_dim * 8, 4, 2, 1),
            nn.Conv2d(feature_dim * 8, 1, 4, 2, 0),
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, z_dim, no_channels, feature_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            single_block_gen(z_dim, feature_dim*16, 4, 1, 0),
            single_block_gen(feature_dim*16, feature_dim*8, 4, 2, 1),
            single_block_gen(feature_dim*8, feature_dim*4, 4, 2, 1),
            single_block_gen(feature_dim*4, feature_dim*2, 4, 2, 1),
            single_block_gen(feature_dim*2, no_channels, 4, 2, 1, True),
        )

    def forward(self, x):
        return self.generator(x)
