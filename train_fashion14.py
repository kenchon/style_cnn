# -*- coding: utf-8 -*-

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
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

dir = '/home/hondoh/source/FashionStyle14_v1/'
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
        label = labels.index(match[0])

        image = Image.fromarray(image)
        image = image.convert("RGB")

        # TEMP: in order to check the converted image
        """
        if ".png" in img_path:
            plt.imshow(image)
            plt.show()
        """
        if self.transform:
            image = self.transform(image)   # transform image to tensor


        #print(image.shape, img_path)

        sample = {'image':image, 'label':label}
        return sample

def compute_mean_std():
    transform = transforms.Compose(
        [#transforms.ToPILImage(),
         transforms.Resize((384, 256)),
         transforms.ToTensor()])
    dataset = FashionStyle14Dataset(dir + "whole_list.csv", dir, transform = transform)


    mean = 0.
    std = 0.
    nb_samples = 0.
    loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False)

    for i,data in enumerate(loader):
        data = data['image']
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        if i % 1000 == 0:
            print(str(i) + "th round")

    mean /= nb_samples
    std /= nb_samples

    print(mean, std)

def main():
    transform = transforms.Compose(
        [#transforms.ToPILImage(),
         transforms.Resize((384, 256)),
         transforms.ToTensor(),
         transforms.Normalize((0.6408, 0.6052, 0.5831), (0.2363, 0.2406, 0.2397))])

    # Loading dataset
    train_set = FashionStyle14Dataset(dir + "train.csv", dir, transform = transform)
    val_set = FashionStyle14Dataset(dir + "valid.csv", dir, transform = transform)
    test_set = FashionStyle14Dataset(dir + "test.csv", dir, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32,shuffle=False)

    print(train_set.__len__())
    print(test_set.__len__())
    print(val_set.__len__())

    # Training Settings
    epochs = 1

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):

            #input_batch, label_batch = data
            print(data['image'].shape)

if __name__ == "__main__":
    compute_mean_std()
