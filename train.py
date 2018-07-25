import torch
from torch import exp
from torch.autograd import Variable
from torch import autograd
from torchvision import models, transforms
from PIL import Image

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

import subprocess as sp
from stylenet import Stylenet
from stylenet import modelB
import image_sampling
from image_sampling import triplet_sampling
#import validate
#import gc
#import random
#import csv

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def triplet_loss(feat, sim):
    alpha = 0.0001

    dist_pos = torch.norm(feat[0] - feat[1], 2)
    dist_neg = torch.norm(feat[0] - feat[2], 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    loss = (reg_dist_pos)**2 - (1-sim[0])*alpha*reg_dist_pos

    return loss


def classfi_loss(y, target):
    N = np.sum(target)
    for y_i,t in zip(y, target):
        loss += -t/N*log(y_i)
    return loss


"""
learning
"""
model = stylenet.modelB
model = model.cuda()

learning_rate = 0.001
epochs = 3
optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate)
batch_size = 24

for epoch in range(epochs):
    f_sim = open("./triplet_v3_r.csv","r")
    reader_csv = csv.reader(f_sim)
    row = []

    for k,i in enumerate(reader_csv):
        row.append(i)

    lines = list(range(len(row)))
    random.shuffle(lines)

    batchs = []
    idx = 0

    for i in range( int(len(row)/batch_size) ):
        batchs.append(lines[idx: idx + batch_size])
        idx += batch_size
    print("int(size/batch_size) = {}".format(int(len(row)/batch_size)))
    print(len(batchs))

    for o, batch in enumerate(batchs[:2000]):

        loss = 0
        model.train(True)

        # img, sim are list where i th column holds i th triplet
        img, sim, pred = triplet_sampling(row, batch)

        batch_count = 0

        feat = []
        for j in range(3):
            feat.append(model.forward(Variable(img[j]).cuda()))
        for i in range(batch_size):
            # herein the loss is computed
            loss += triplet_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]))
            loss += classfi_loss(feat[2][i],target[i])
            batch_count += 1

        loss = loss/batch_size
        print(o, loss.data[0])

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "cond2_{}.pth".format(epoch))

