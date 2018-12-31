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

import argparse
import pprint

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

dir = '/home/hondoh/source/FashionStyle14_v1/'
labels = ["conservative","dressy","ethnic","fairy","feminine","gal","girlish",
            "kireime-casual","lolita", "mode","natural","retro","rock","street"]
transform = transforms.Compose(
    [#transforms.ToPILImage(),
     transforms.Resize((384, 256)),
     transforms.ToTensor(),
     transforms.Normalize((0.6408, 0.6052, 0.5831), (0.2363, 0.2406, 0.2397))])
N_labels = len(labels)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description = 'Train StyleNet on FashionStyle14')
    parser.add_argument('--lr', dest='lr',
                        default=1e-3, type=float)
    parser.add_argument('--epochs', dest='epochs',
                        default=10, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=32, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        default=0.5, type=float)
    parser.add_argument('--use_finetuned', dest='use_finetuned',
                        default=False, type=bool)

    args = parser.parse_args()
    return args

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

    def load_params(params_path):
        weight_dict = torch.load(weight_path)

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

def load_weight(model, weight_path):
    weight = torch.load(weight_path)
    state_dict =  model.state_dict()
    for key in state_dict:
        if "linear2" not in key:
            state_dict[key] = weight[key]
            print("load weight: key = " + key)
    model.load_state_dict(state_dict)
    return model

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

def test(model):
    model.cuda()

    test_set = FashionStyle14Dataset(dir + "test.csv", dir, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=False)
    class_correct = list(0. for i in range(N_labels))
    class_total = list(0. for i in range(N_labels))

    with torch.no_grad():
        for data in test_loader:
            input = data['image'].cuda()
            label = data['label'].cuda()

            outputs = model(input)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == label).squeeze()
            for i in range(len(label)):
                l = label[i]
                class_correct[l] += c[i].item()
                class_total[l] += 1

    sum_correct = 0
    sum_total   = 0
    for i in range(N_labels):
        print('Accuracy of %5s : %.2f %%' % (
            labels[i], 100 * class_correct[i] / class_total[i]))
        sum_correct += class_correct[i]
        sum_total   += class_total[i]
    print("Average:\t %.2f %%" % (100 * sum_correct / sum_total))

def get_valid_loss(model):
    model.cuda()

    val_set = FashionStyle14Dataset(dir + "valid.csv", dir, transform = transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            input = data['image'].cuda()
            label = data['label'].cuda()

            outputs = model(input)
            loss += criterion(outputs, label)

    return loss/val_set.__len__()

def main():
    args = parse_args()
    pprint.pprint(vars(args))

    # Loading dataset
    train_set = FashionStyle14Dataset(dir + "train.csv", dir, transform = transform)
    val_set = FashionStyle14Dataset(dir + "valid.csv", dir, transform = transform)
    test_set = FashionStyle14Dataset(dir + "test.csv", dir, transform = transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,shuffle=True)

    print(train_set.__len__())
    print(test_set.__len__())
    print(val_set.__len__())

    # Loading Model
    model = Stylenet()
    if args.use_finetuned:
        model_path = "./linear_weight_softmax_pro_ver_162000_.pth"
        model = load_weight(model, model_path)
    else:
        model_path = "./stylenet14_pretrain.pth"
        model.load_state_dict(torch.load(model_path))

    model.train(True)
    model.cuda()

    # Training Settings
    epochs = args.epochs
    lr = args.lr
    lr_decay_gamma = args.lr_decay_gamma

    criterion = nn.CrossEntropyLoss()
    model_save_path = "./stylenet14_pretrain.pth"

    print("done")

    last_valid_loss = 1e10

    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        running_loss = 0.0
        print("="*5 + "Epoch " + str(epoch + 1) + "="*5)

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
            #print('[%d, %5d] loss: %.3f' %
            #    (epoch + 1, i + 1, loss))

        # learning rate decay criterion
        valid_loss = get_valid_loss(model).item()
        training_loss = running_loss/train_set.__len__()

        print(f"valid loss: {valid_loss}")
        print(f"training loss: {training_loss}")
        if (0.98 < last_valid_loss/valid_loss < 1.02) :
            lr = lr * lr_decay_gamma
            print(f"Learning rate decay applied: lr = {lr}")
        last_valid_loss = valid_loss

        torch.save(model.state_dict(), model_save_path)

        # training convergence criterion
        if lr < 5e-8:
            break

    test(model)

if __name__ == "__main__":
    main()
    model = Stylenet()
    model_path = "./stylenet14_pretrain.pth"
    model.load_state_dict(torch.load(model_path))
    test(model)
