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
import load_model
import image_sampling
import test
from image_sampling import triplet_sampling
import compute_similarity as cs
import mymodules.line_notify as ln
#import validate
#import gc
import random
import csv

w = cs.weights

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

    loss_t = (reg_dist_pos)**2 - (1-sim[0])*alpha*reg_dist_pos

    return loss_t


def classfi_loss(log_y, target):
    loss_c = 0
    sum_w = 0
    #N = len(target)
    for t in target:
        sum_w += w[t]
        loss_c += -w[t]*log_y[t]   # cross entropy
    loss_c = loss_c/(sum_w)
    return loss_c


if __name__ == "__main__":
    model = load_model.model
    model = model.cuda()

    # learning settings
    learning_rate = 0.001
    epochs = 1000
    optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate)
    batch_size = 16

    for epoch in range(epochs):
        f_sim = open("./triplet.csv","r")
        reader_csv = csv.reader(f_sim)
        row = []

        for k,i in enumerate(reader_csv):
            row.append(i)

        lines = list(range(len(row)))
        random.shuffle(lines)

        #row = row[:1235520]

        batchs = []
        idx = 0
        alpha_c = 0.01

        for i in range( int(len(row)/batch_size) ):
            batchs.append(lines[idx: idx + batch_size])
            idx += batch_size

        #print("int(size/batch_size) = {}".format(int(len(row)/batch_size)))
        print(len(batchs))

        for o, batch in enumerate(batchs):
            loss = 0
            model.train(True)

            # img, sim are list where i th column holds i th triplet

            #img, sim, pred = triplet_sampling(row, batch)
            img, sim, target = triplet_sampling(row, batch, do_classification = True)

            batch_count = 0

            feat = []
            for j in range(3):
                feat.append(model.forward(Variable(img[j]).cuda()))
            for i in range(batch_size):
                # herein the loss is computed
                loss += triplet_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]))
                loss += alpha_c * classfi_loss(feat[2][i],target[i])
                batch_count += 1

            del feat
            loss = loss/batch_size
            #print(o, loss.cpu().detach().numpy())

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if(o!= 0 and o%500 == 0):
                model_path = "./result/params/prams_lr001_clas=True_epoch{}_iter{}_3_.pth".format(epoch, o)
                torch.save(model.state_dict(), model_path)
                mess = test.test2(model_path)
                #ln.notify(mess)
                print(mess)
