from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import re

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

labels = ["conservative","dressy","ethnic","fairy","feminine","gal","girlish",
            "kireime-casual","lolita", "mode","natural","retro","rock","street"]

class FashionStyle14Dataset(Dataset):
    """FashionStyle14 dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.img_paths = pd.read_csv(csv_file, sep = '\n')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir,
                                self.img_paths.iloc[idx, 0])
        image = io.imread(img_path)
        match = re.findall('dataset/(.*)/', img_path)
        label = match[0]

        if self.transform:
            image = self.transform(image)   # transform image to tensor

        sample = {'image':image, 'label':label}
        return sample

if __name__ == "__main__":

    dir = '/home/hondoh/source/FashionStyle14_v1/'
    train_set = FashionStyle14Dataset(dir + "train.csv", dir)
    val_set = FashionStyle14Dataset(dir + "val.csv", dir)
    test_set = FashionStyle14Dataset(dir + "test.csv", dir)

    print(train_set.__len__())
    print(test_set.__len__())
    print(val_set.__len__())
