import os

from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd

class PainDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(PainDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
        image = read_image(img_path)

        label = self.img_labels.iloc[i, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)
