import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class UTKDatasetAge(Dataset):
    """Custom Dataset for loading UTKFace images"""

    def __init__(self, csv_path, img_dir, num_classes, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.df = df
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.df.iloc[index]['filename']))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]
    
    
class UTKDatasetAgeBinary(Dataset):
    """Custom Dataset for loading UTKFace images"""

    def __init__(self, csv_path, img_dir, num_classes, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.df = df
        self.y = df['age'].values
        self.transform = transform

        ###################################
        # New:
        self.num_classes = num_classes
        ###################################

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.df.iloc[index]['filename']))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        ########################################################
        # New:
        levels = [1]*label + [0]*(self.num_classes - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)
        ########################################################

        return img, label, levels

    def __len__(self):
        return self.y.shape[0]