# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import re
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

dir = '/home/hondoh/source/FashionStyle14_v1/'
labels = ["conservative","dressy","ethnic","fairy","feminine","gal","girlish",
            "kireime-casual","lolita", "mode","natural","retro","rock","street"]

N_labels = len(labels)

class Stylenet(nn.Module):
    def __init__(self):
        super(Stylenet, self).__init__()
        self.relu = nn.ReLU
        self.conv1 = nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1))
        self.conv2 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1))
        self.conv2_drop = nn.Dropout(0.25)
        self.pool1 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn1 = nn.BatchNorm2d(64,0.001,0.9,True)
        self.conv3 = nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1))
        self.conv4 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1))
        self.conv4_drop = nn.Dropout(0.25)
        self.pool2 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn2 = nn.BatchNorm2d(128,0.001,0.9,True)
        self.conv5 = nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1))
        self.conv6 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.conv6_drop = nn.Dropout(0.25)
        self.pool3 =nn.MaxPool2d((4, 4),(4, 4))
        self.bn3 = nn.BatchNorm2d(256,0.001,0.9,True)
        self.conv7 = nn.Conv2d(256,128,(3, 3),(1, 1),(1, 1))
        self.linear1 = nn.Linear(3072,128)
        self.linear2 = nn.Linear(128, N_labels)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.conv2_drop(x)
        x = self.bn1(self.pool1(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv4_drop(x)
        x = self.bn2(self.pool2(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.conv6_drop(x)
        x = self.bn3(self.pool3(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1,3072)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.logsoftmax(x)
        return  x

class FashionStyle14Dataset(Dataset):
    """FashionStyle14 dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.img_paths = pd.read_csv(csv_file, sep = ',')
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

    # Loading Model
    model = Stylenet()
    model.train(True)
    model.cuda()

    # Training Settings
    epochs = 1
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            input = data['image'].cuda()
            label = data['label'].cuda()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, loss))
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

if __name__ == "__main__":
    main()
