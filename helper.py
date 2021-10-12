import torch
import yaml
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


def device_selector():
    return "cuda" if torch.cuda.is_available() else "cpu"
    # return "cpu"


def view_image(image):
    plt.figure()
    plt.imshow(image, cmap='Greys')


def test(disc, gen):
    N, in_channels, H, W = 8, 1, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    print(f'Discriminator Output Shape :{disc(x).shape}')
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print(f'Generator Output Shape :{gen(z).shape}')


def parameter_reader():
    yml_file = open('./config.yml')
    parsed_yml = yaml.load(yml_file, Loader=yaml.FullLoader)
    network_parameter = parsed_yml['network_parameter']
    training_parameter = parsed_yml['training_parameter']
    transform_parameter = parsed_yml['transform_parameter']
    return network_parameter, training_parameter, transform_parameter


def weight_initializer(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def show_summary(real_images, generated_images, step):

    real_writer = SummaryWriter(f'logs/real_images')
    generated_writer = SummaryWriter(f'logs/generated_images')
    real_image_grid = torchvision.utils.make_grid(
        real_images[:32], normalize=True
    )
    generated_image_grid = torchvision.utils.make_grid(
        generated_images[:32], normalize=True
    )
    real_writer.add_image("Real Images", real_image_grid, global_step=step)
    generated_writer.add_image("Generated Images", generated_image_grid, global_step=step)

