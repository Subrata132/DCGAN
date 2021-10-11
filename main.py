from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import ImageDataset


def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = ImageDataset(root_dir='./data/', transform=transform)
    data_loader = DataLoader(dataset,
                             batch_size=64,
                             shuffle=True)


if __name__ == '__main__':
    main()