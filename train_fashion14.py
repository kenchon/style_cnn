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
import mymodules.line_notify as ln
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

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
                        default=50, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        default=32, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        default=0.5, type=float)
    parser.add_argument('--use_finetuned', dest='use_finetuned',
                        default=False, type=bool)
    parser.add_argument('--dropout_p', dest='dropout_p',
                        default=0.5, type=float)
    parser.add_argument('--weight_decay_lambda', dest='weight_decay_lambda',
                        default=0, type=float)
    parser.add_argument('--do_test', dest='do_test',
                        default=False, type=bool)
    parser.add_argument('--method', dest='method',
                        default='ours', type=str)
    parser.add_argument('--lr_decay_criterion', dest='lr_decay_criterion',
                        default='validation_error', type=str)

    args = parser.parse_args()
    return args

class Stylenet(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(Stylenet, self).__init__()
        self.relu = nn.ReLU
        self.conv1 = nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1))
        self.conv2 = nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1))
        self.conv2_drop = nn.Dropout(p=dropout_p)
        self.pool1 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1))
        self.conv4 = nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1))
        self.conv4_drop = nn.Dropout(p=dropout_p)
        self.pool2 = nn.MaxPool2d((4, 4),(4, 4))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1))
        self.conv6 = nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1))
        self.conv6_drop = nn.Dropout(p=dropout_p)
        self.pool3 =nn.MaxPool2d((4, 4),(4, 4))
        self.bn3 = nn.BatchNorm2d(256)
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
            #print("load weight: key = " + key)
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
    model.train(False)

    test_set = FashionStyle14Dataset(dir + "test.csv", dir, transform = transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32,shuffle=True)
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

def get_acc(model, phase):
    model.cuda()
    model.train(False)

    data_set = FashionStyle14Dataset(dir + f"{phase}.csv", dir, transform = transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=32,shuffle=True)
    class_correct = list(0. for i in range(N_labels))
    class_total = list(0. for i in range(N_labels))

    with torch.no_grad():
        for data in data_loader:
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
        sum_correct += class_correct[i]
        sum_total   += class_total[i]
    total_acc = 100 * sum_correct / sum_total
    return total_acc

def get_loss(model, phase):
    """
    Returns loss(float) for {train|valid|test} set.
    Args:   model(nn.Module)...network to measure the loss
            phase(str)...train, valid, test
    """
    model.cuda()
    model.train(False)

    data_set = FashionStyle14Dataset(dir + f"{phase}.csv", dir, transform = transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=32,shuffle=False)
    criterion = nn.CrossEntropyLoss()
    loss = 0.0

    with torch.no_grad():
        for data in data_loader:
            input = data['image'].cuda()
            label = data['label'].cuda()

            outputs = model(input)
            loss += criterion(outputs, label)

    return loss/data_set.__len__()

def train(args):

    # Loading dataset
    train_set = FashionStyle14Dataset(dir + "train.csv", dir, transform = transform)
    val_set = FashionStyle14Dataset(dir + "valid.csv", dir, transform = transform)
    test_set = FashionStyle14Dataset(dir + "test.csv", dir, transform = transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,shuffle=True)

    # Loading Model
    model = Stylenet(args.dropout_p)
    if args.use_finetuned:
        if args.method == 'ours':
            model_path = "./linear_weight_softmax_pro_ver_77000_newweight.pth"
            model_path = "./linear_weight_softmax_pro_ver_162000_.pth"
            #model_path = "./linear_weight_softmax_datasetsize_11999_0.2.pth"
            #model_path = "./result/params/prams_lr001_clas=False_pre_epoch0_iter55000_12.pth"
            print(f"weight: {model_path}")
        if args.method == 'imagenet':
            model_path = "./imagenet_0.pth"
            print("use imagenet-pretrained model")
        else:
            #model_path = "./result/params/prams_lr001_clas=False_pre_epoch0_iter13200_11.pth"
            model_path = "KLtrue_63999_1.pth"
        model = load_weight(model, model_path)

    model.train(True)
    model.cuda()

    # Training Settings
    epochs = args.epochs
    lr = args.lr
    lr_decay_gamma = args.lr_decay_gamma

    criterion = nn.CrossEntropyLoss()
    #model_save_path = "./stylenet14_pretrain.pth"

    last_valid_loss = 1e10
    last_valid_acc  = 0.0
    patience = 0
    loss_dict = {"train":[], "valid":[], "test":[]}
    valid_accs = []

    for epoch in range(epochs):
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay_lambda)
        running_loss = 0.0
        print("="*10 + "Epoch " + str(epoch + 1) + "="*10)

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

        # compute loss for each phase
        valid_loss = get_loss(model, phase="valid").item()
        loss_dict["valid"].append(valid_loss)
        test_loss = get_loss(model, phase="test").item()
        loss_dict["test"].append(test_loss)
        train_loss = running_loss/train_set.__len__()
        loss_dict["train"].append(train_loss)

        # pict and send the graph of loss progress
        for phase in ["train", "valid", "test"]:
            plt.plot(np.array(loss_dict[phase]))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("Loss Progress")
        path2img = "loss_progress.png"
        plt.savefig(path2img)
        if(epoch%5 == 4):
            ln.send_image_(path2img)

        valid_acc = get_acc(model, phase="valid")
        valid_accs.append(valid_acc)

        print(f"training loss: \t{train_loss:.6f}")
        print(f"valid loss: \t{valid_loss:.6f} \tvalid acc. \t{valid_acc:.6f}")
        print(f"test loss: \t{test_loss:.6f}")

        if(args.lr_decay_criterion == "validation_error"):
            if (last_valid_acc/valid_acc > 1):
                lr = lr * lr_decay_gamma
                #model.load_state_dict(torch.load(f"./stylenet14_pretrain_{epoch-1}_teian_.pth"))
                print(f"Learning rate decay applied: lr = {lr}")

            last_valid_acc = valid_acc
        else:
            if epoch+1 % 10 == 0:
                lr = lr * lr_decay_gamma
                print(f"Learning rate decay applied: lr = {lr}")

        model_save_path = f"./stylenet14_{epoch}_{args.method}.pth"
        torch.save(model.state_dict(), model_save_path)

        # training convergence criterion
        if lr < 1e-6:
            break

    test(model)

if __name__ == "__main__":
    args = parse_args()
    print("="*10+"SETTINGS"+"="*10)
    pprint.pprint(vars(args))

    if args.do_test:
        model = Stylenet()
        #test_model_path = f"./stylenet14_pretrain_20_{args.method}_lr5e-3.pth"
        test_model_path = "./stylenet14_pretrain_8_wo_finetune.pth"
        test_model_path = "stylenet14_imagenet_16_naive.pth"
        test_model_path = "stylenet14_pretrain_99_kizon.pth"
        #test_model_path = "stylenet14_imagenet_29_ours.pth"
        #test_model_path = "stylenet14_imagenet_18_ours.pth"
        test_model_path = "stylenet14_pretrain_15_kizon_.pth" # naive
        #test_model_path = "stylenet14_imagenet_10_ours.pth" # state_of_the_art
        #test_model_path = "stylenet14_imagenet_16_naive.pth"
        test_model_path = "stylenet14_imagenet_9_imagenet.pth"
        print(f"test the model: {test_model_path}")
        model.load_state_dict(torch.load(test_model_path))
        test(model)
    else:
        train(args)
