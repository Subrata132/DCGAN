import cv2
from PIL import Image
from os import listdir
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(listdir(self.root_dir), key=len)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(self.root_dir+self.image_files[index], 0)
        image = Image.fromarray(image)
        if self.transform:
            return self.transform(image)
        else:
            return image

