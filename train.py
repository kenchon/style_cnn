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

def triplet_loss(feat, sim, use_proposed = False):
    alpha = 0.01

    dist_pos = torch.norm(feat[0] - feat[1], 2)
    dist_neg = torch.norm(feat[0] - feat[2], 2)

    reg_dist_pos = exp(dist_pos)/(exp(dist_pos) + exp(dist_neg))
    reg_dist_neg = exp(dist_neg)/(exp(dist_pos) + exp(dist_neg))

    if(use_proposed):
        loss_t = (((1 - sim[0])*alpha - reg_dist_pos)**2 + (0 - reg_dist_neg)**2)/2
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

def save_loss(loss_prog, temp_score, add = ""):
    f_loss = open('loss_progress.txt', mode = 'a')
    for i in loss_prog:
        f_loss.write(str(i)+'\n')
    f_loss.close()

    f_acc = open('acc_progress.txt', mode = 'a')
    f_acc.write(str(temp_score)+'\n')
    f_acc.close()

def get_loss_acc_fig(loss_prog, temp_score):
    save_loss(loss_prog, temp_score)
    path_to_fig = get_prog_image()
    return path_to_fig


def get_prog_image(add = "", split = 100):
    with open('loss_progress.txt','r') as f:
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

    with open('acc_progress.txt','r') as f:
        lines = f.readlines()

    acc_list = []
    for line in lines:
        acc_list.append(float(line.strip()))

    X = [i for i in range(int(len(mean)/split)+1)]
    X = np.array(X)*split

    fig, (axU, axD) = plt.subplots(2, 1)

    c = 'blue'
    axU.plot(mean, color = c)
    axU.grid(True)
    axU.set_title('Loss Progress')

    axD.plot(acc_list, color = c)
    axD.grid(True)
    axD.set_title('Acc. Progress')

    path_to_img = cur_path + 'loss_progress.png'
    fig.savefig(path_to_img)

    return path_to_img

def load_training_ids(batch_size, use_proposed):
    f_sim_path = "./triplet.csv" if(use_proposed) else "./triplet_pre.csv"
    print(f_sim_path)
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
    do_retrain = False

    # load model
    if do_classification and not do_retrain:
        model = stylenet.Stylenet()
        #model.load_state_dict(torch.load('./experience_result/linear_weight_softmax_118000.pth'))
        model.load_state_dict(torch.load('linear_weight_softmax_notpro_37000.pth'))
        forward = model.forward_
    elif do_retrain and do_classification:
        model = stylenet.get_model()
        model = insert_trained_weight(model)
        forward = model.forward_
    elif do_retrain:
        model = load_model.model
        model.load_state_dict(torch.load('stylenet.pth'))
        forward = model.forward

    do_classification = False
    model.cuda()
    model.train()

    # learning settings
    learning_rate = 1e-2
    epochs = 2000
    optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    batch_size = 22
    alpha_c = 0.01
    split = 1000

    max_score = 0
    loss_progress = []

    for epoch in range(epochs):

        row, batchs = load_training_ids(batch_size, use_proposed = True)
        print(len(batchs))

        for o, batch in enumerate(batchs):
            loss = 0
            model.train()

            if(o% split == 0 and o != 0):
                model_path = "./result/params/prams_lr001_clas=False_pre_epoch{}_iter{}_11.pth".format(epoch, o)
                optim_path = "./result/params/optim_lr001_clas=False_pre_epoch{}_iter{}_11.pth".format(epoch, o)
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optim_path)

                temp_score = test.test(model_path, do_classification, model)

                path_to_fig = get_loss_acc_fig(loss_progress, temp_score)
                ln.send_image_(path_to_fig, 'loss progress')
                ln.notify('{} {}'.format(str(temp_score), model_path))

                loss_progress = []

            # img, sim are list: i th column holds i th triplet
            img, sim, target, target_npy = triplet_sampling(row, batch)
            batch_count = 0

            feat = []
            pred = []
            for j in range(3):
                f,p = forward(Variable(img[j]).cuda())
                if(do_classification and j == 2):
                    pred.append(p)

                feat.append(f)

            # LOSS COMPUTING
            for i in range(batch_size):
                loss += triplet_loss([feat[0][i],feat[1][i],feat[2][i]], (sim[0][i],sim[1][i]), use_proposed)
                if(do_classification):
                    loss += alpha_c * binary_classfi_loss(pred[0][i], target_npy[i], use_proposed)

            loss = loss/batch_size
            print("{} {:.4f}".format(o, loss.cpu().detach().numpy()))
            loss_progress.append("{:.4f}".format(loss.cpu().detach().numpy()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
