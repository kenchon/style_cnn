import torch
from torch import exp
from torch.autograd import Variable
from torch import autograd
from torchvision import models, transforms
from PIL import Image

import numpy as np
from sklearn import svm

import subprocess as sp
import os
import matplotlib.pyplot as plt
import load_model
import stylenet
import image_sampling
import test
from image_sampling import triplet_sampling
import compute_similarity as cs
import mymodules.line_notify as ln
#import validate
#import gc
import random
import csv

cur_path = os.getcwd()
sum_w = cs.sum_w
print(sum_w , 'sum_w')
N_tags = 66

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def triplet_loss(feat, sim, use_proposed = True):
    alpha = 0.01

    dist_pos = torch.norm(feat[0] - feat[1], 2)
    dist_neg = torch.norm(feat[0] - feat[2], 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    if(use_proposed):
        loss_t = (((1 - sim[0])*alpha - reg_dist_pos)**2)
        return loss_t
    else:
        return (reg_dist_pos)**2

def triplet_cross_entropy_loss(feat, sim, use_proposed = True):
    alpha = 0.1
    w1 = 0.8
    w2 = 1 - w1

    dist_pos = torch.norm(feat[0] - feat[1], 2)
    dist_neg = torch.norm(feat[0] - feat[2], 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    loss_t = -alpha*(1-sim[0])*torch.log(reg_dist_pos)*w1 -1*torch.log(reg_dist_neg)*w2
    return loss_t/2

def classfi_loss(log_y, target, use_proposed = True):
    loss_c = 0

    if(use_proposed):
        sum_w = 0
        for t in target:
            sum_w += w[t]
            loss_c += -w[t]*log_y[t]   # cross entropy
        loss_c = loss_c/(sum_w)
        return loss_c

    else:
        N = len(target)
        for t in target:
            loss_c += -1*log_y[t]   # cross entropy
        loss_c = loss_c/N
        return loss_c

def binary_classfi_loss(conf, target_npy, use_proposed = True):
    loss_c = 0
    if(use_proposed):
        for t in range(N_tags):
            y_k = (target_npy[t] - 0.5) * 2
            loss_c += w[t]*torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy
        return loss_c/sum_w

    else:
        for t in range(N_tags):
            y_k = (target[t] - 0.5) * 2
            loss_c += torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy
        return loss_c/N_tags

def insert_trained_weight(model, path_to_weight = "linear_weight.pth"):

    weight_dict = torch.load(path_to_weight)
    layers = ['linear2.weight', 'linear2.bias']
    state_dict = model.state_dict()

    for layer in layers:
        state_dict[layer] = weight_dict[layer]
    model.load_state_dict(state_dict)

    return model

def save_loss(loss_prog, add = ""):
    #with open('loss_progress.txt', mode = 'a') as f_loss:
    #    f_loss.write('\n'.join(loss_prog))
    f_loss = open('loss_progress'+add+'.txt', mode = 'a')
    for i in loss_prog:
        f_loss.write(str(i)+'\n')
    f_loss.close()

def send_prog_image(add = "", split = 100):
    with open('loss_progress'+add+'.txt','r') as f:
        lines = f.readlines()

    num_split = split
    sum_e = 0
    mean = []

    for i in range(len(lines)):
        e =float( lines[i].strip() )
        sum_e += e
        if((i+1)%num_split == 0):
            mean.append(sum_e/num_split)
            sum_e = 0

    plt.plot(mean)
    path_to_img = cur_path + 'loss_progress.png'
    plt.savefig(path_to_img)

    return path_to_img

def load_training_ids(batch_size, use_proposed):
    f_sim_path = "./triplet.csv" if(use_proposed) else "./triplet_pre.csv"
    f_sim      = open(f_sim_path,"r")
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

    return row, batchs

if __name__ == "__main__":
    w = cs.weights
    use_proposed = True
    do_classification = True

    # load model
    if do_classification:
        model = stylenet.Stylenet()
        model.load_state_dict(torch.load('linear_weight_softmax_118000.pth'))
        forward = model.forward_
    else:
        model = load_model.model
        model.load_state_dict(torch.load('stylenet.pth'))
        forward = model.forward

    model.cuda()
    model.train()

    # learning settings
    learning_rate = 1e-4
    epochs = 500
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad = True)
    batch_size = 22
    alpha_c = 1
    split = 500

    max_score = 0
    loss_progress = []

    #model.load_state_dict(torch.load('./result/params/prams_lr001_clas=True_epoch0_iter10000_5.pth'))
    #optimizer.load_state_dict(torch.load('./result/params/optim_lr001_clas=True_epoch0_iter10000_5.pth'))

    for epoch in range(epochs):

        row, batchs = load_training_ids(batch_size, use_proposed)
        print(len(batchs))

        for o, batch in enumerate(batchs):
            loss = 0
            model.train()

            if(o% split == 0 and o != 0):
                save_loss(loss_progress)
                loss_progress = []
                path_to_fig = send_prog_image()
                ln.send_image_(path_to_fig, 'loss progress')

                model_path = "./result/params/prams_lr001_clas=True_epoch{}_iter{}_7.pth".format(epoch, o)
                optim_path = "./result/params/optim_lr001_clas=True_epoch{}_iter{}_7.pth".format(epoch, o)
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)

                temp_score = test.test(model_path, do_classification)
                ln.notify("{} {}".format(str(temp_score), model_path))

                print("{} {}".format(temp_score, max_score))



            # img, sim are list: i th column holds i th triplet
            img, sim, target, target_npy = triplet_sampling(row, batch)
            batch_count = 0

            feat = []
            pred = []
            for j in range(3):
                if do_classification:
                    f, p = forward(Variable(img[j]).cuda())
                    if(j == 2):
                        pred.append(p)
                else:
                    f = forward(Variable(img[j]).cuda())
                feat.append(f)

            # LOSS COMPUTING
            for i in range(batch_size):
                # loss += triplet_cross_entropy_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]), use_proposed)
                loss += triplet_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]), use_proposed)
                if(do_classification):
                    loss += alpha_c * binary_classfi_loss(pred[0][i], target_npy[i], use_proposed)

                #batch_count += 1

            loss = loss/batch_size
            print("{} {:.4f}".format(o, loss.cpu().detach().numpy()))
            loss_progress.append("{:.4f}".format(loss.cpu().detach().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
