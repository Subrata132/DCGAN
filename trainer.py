import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from model import Discriminator, Generator
from helper import parameter_reader, device_selector, weight_initializer, show_summary
from data_loader import ImageDataset


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


def train():
    # Read Parameters
    network_parameter, training_parameter, transform_parameter = parameter_reader()

    # Data Preparation
    transform = transforms.Compose(
        [
            transforms.Resize(transform_parameter['image_size']),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )
    dataset = ImageDataset(
        root_dir=network_parameter['root_dir'],
        transform=transform
    )
    data_loader = DataLoader(dataset=dataset, batch_size=training_parameter['batch_size'], shuffle=True)

    # Device Selection
    device = device_selector()

    # Model Setup
    discriminator = Discriminator(no_channels=network_parameter['no_channel'],
                                  feature_dim=network_parameter['disc_feature']).to(device)
    weight_initializer(discriminator)
    generator = Generator(z_dim=network_parameter['z_dim'],
                          no_channels=network_parameter['no_channel'],
                          feature_dim=network_parameter['gen_feature']).to(device)
    weight_initializer(generator)

    # Optimizer and Loss function
    gen_opt = optim.Adam(generator.parameters(),
                         lr=training_parameter['learning_rate'],
                         betas=(training_parameter['beta1'], training_parameter['beta2']))
    dis_opt = optim.Adam(discriminator.parameters(),
                         lr=training_parameter['learning_rate'],
                         betas=(training_parameter['beta1'], training_parameter['beta2']))
    criterion = nn.BCELoss()
    generator.train()
    discriminator.train()

    # Training Loop
    step = 0
    for epoch in range(training_parameter['epochs']):
        for batch_idx, real_image in enumerate(data_loader):

            # Take real image and create fake image
            real_image = real_image.to(device)
            noise = torch.randn((training_parameter['batch_size'], network_parameter['z_dim'], 1, 1)).to(device)
            fake_image = generator(noise)

            # Train Discriminator
            disc_real = discriminator(real_image).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake_image).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2
            discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            dis_opt.step()

            # Train Generator
            output = discriminator(fake_image).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            generator.zero_grad()
            loss_gen.backward()
            gen_opt.step()

            # Print Summary
            if batch_idx % 100 == 0 and batch_idx > 0:
                generator.eval()
                discriminator.eval()
                print(
                    f"Epoch [{epoch}/{training_parameter['epochs']}] Batch {batch_idx}/{len(data_loader)} \
                              Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )