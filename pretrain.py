import torch
import torch.nn as nn

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import mymodules.line_notify as ln
from collections import OrderedDict as od
from math import log, exp

import load_model
import compute_similarity as cs
import image_sampling as smp
import train as tr
import stylenet
import test
import json

cur_path = os.getcwd()
N_tags = 66
il_list = cs.il_list        # ilegal photo ids
SIZE = cs.SIZE
w = cs.weights
label_array = np.load("./noisy.npy")     # load numpy based labels
use_proposed = True
batch_size = 48
sum_w = cs.sum_w
delt = 0

def classfi_loss(conf, target, use_proposed = True):
    loss_c = 0
    if(use_proposed):
        for t in range(N_tags):
            y_k = (target[t] - 0.5) * 2
            loss_c += (w[t]**0.5)*torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy

        return loss_c/sum_w

    else:
        for t in range(N_tags):
            y_k = (target[t] - 0.5) * 2
            loss_c += torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy
        return loss_c/N_tags

def classfi_loss_with_binary_output(conf, target, use_proposed = True):
    """
    E.Simo-Serra+,CVPR2016 eq.(7) for details
    """
    loss_c = 0.0
    if(use_proposed):
        for t in range(N_tags):
            i0 = t*2
            i1 = t*2 + 1
            loss_c += (w[t]**2)*(-int(target[t]) + torch.log( torch.exp(conf[i0]) + torch.exp(conf[i1])))

        return loss_c/sum_w
    else:
        for t in range(N_tags):
            x0 = t*2
            x1 = t*2 + 1
            loss_c += (-target[t] + torch.log( torch.exp(conf[x0]) + torch.exp(conf[x1])))
        return loss_c/N_tags

def classfi_loss_softmax(conf, target, use_proposed = True):
    loss_c = 0
    target_ = np.where(np.array(target) == 1)[0]
    for t in target:
        t = int(t)
        loss_c += -w[t]*conf[t]
    return loss_c/sum_w

def get_saved_dict(path_to_weight = "linear_weight.pth"):

    dic = torch.load(path_to_weight)
    new_dict = od()
    new_dict['linear2.weight'] = dic['linear1.weight']
    new_dict['linear2.bias'] = dic['linear1.bias']

    return new_dict

def save_loss(loss_prog, temp_score, add = ""):
    f_loss = open('loss_progress_top_half.txt', mode = 'a')
    for i in loss_prog:
        f_loss.write(str(i)+'\n')
    f_loss.close()

    f_acc = open('acc_progress_top_half.txt', mode = 'a')
    f_acc.write(str(temp_score)+'\n')
    f_acc.close()

def get_loss_acc_fig(loss_prog, temp_score):
    save_loss(loss_prog, temp_score)
    path_to_fig = get_prog_image()
    return path_to_fig

def get_prog_image(add = "", split = 100):
    with open('loss_progress_top_half.txt','r') as f:
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

    with open('acc_progress_top_half.txt','r') as f:
        lines = f.readlines()

    acc_list = []
    for line in lines:
        acc_list.append(float(line.strip()))

    X = [i for i in range(int(len(mean)/split)+1)]
    X = np.array(X)*split

    fig, (axU, axD) = plt.subplots(2, 1)

    axU.plot(mean)
    axU.grid(True)
    axU.set_title('Loss Progress')

    axD.plot(acc_list)
    axD.grid(True)
    axU.set_title('Acc. Progress')

    path_to_img = cur_path + 'loss_progress.png'
    fig.savefig(path_to_img)

    return path_to_img

def get_setting():
    with open('learning_config.txt', 'r') as f:
        lines = f.readlines()
    dic = {}
    for l in lines:
        l = l.strip().split(":")
        dic.update({l[0]:float(l[1])})
    return  dic

class LinearUnit(nn.Module):
    def __init__(self):
        super(LinearUnit, self).__init__()
        self.linear2 = nn.Linear(128, N_tags)
        self.sigmoid = nn.Sigmoid()

        self.linear3 = nn.Linear(128, N_classes)
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        x = self.linear2(input)
        x = self.sigmoid(x)

        y = self.linear3(input)
        y = self.softmax(y)
        return x, y

if __name__ == '__main__':
    epochs = 10000000
    split = 1000
    loss_progress = []
    use_ratio = 1

    stopped = 0

    model = stylenet.Stylenet()
    model.load_state_dict(torch.load('pow2_41999_1.pth'))
    #model.load_state_dict(torch.load('./experience_result/linear_weight_softmax_118000.pth'))
    model.cuda()
    model.train()

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.Adadelta(model.parameters(), weight_decay = 5e-4)

    train_list = list(range(0, SIZE))
    random.shuffle(train_list)
    #train_list = train_list[:int(use_ratio*len(train_list))]
    """
    a = [str(i)+"\n" for i in train_list]
    with open(f"trainlist_{use_ratio}.txt", "a") as f:
        f.writelines(a)

    for i, e in enumerate(train_list):
        if(e in il_list):
            new_e = e
            while new_e not in il_list:
                new_e = random.randint(0, SIZE)
            train_list[i] = new_e
    """
    last_loss = 10000

    for epoch in range(stopped, epochs):
        if ((epoch+1)%(split) == 0 and epoch != stopped):

            model_path = 'pow2_{}_{}.pth'.format(epoch, use_ratio)
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), 'optim_softmax.pth')

            temp_score = test.test(model_path, do_classification = True)
            path_to_fig = get_loss_acc_fig(loss_progress, temp_score)
            ln.send_image_(path_to_fig, 'loss progress')
            ln.notify('{} {}'.format(str(temp_score), model_path))

            curr_loss = np.mean(loss_progress)
            """
            if(last_loss/curr_loss < 1):
                last_loss = curr_loss
                learning_rate = 0.1 * learning_rate
                print(f"lr decay applied: lr = {learning_rate}")
                optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
            """
            loss_progress = []

        # image sampling
        tensor = torch.Tensor(batch_size, 3, 384, 256).cuda()

        optimizer.zero_grad()
        #tensor.requires_grad_(False)
        target = list(range(batch_size))         # class labels
        count = 0

        batch_ids = random.sample(train_list, batch_size)
        for i,id in enumerate(batch_ids):
            try:
                tensor[i] = smp.id2tensor(str(id))
                target[i] = label_array[id]
            except:
                id = random.randint(0, SIZE)
                tensor[i] = smp.id2tensor(str(id))
                target[i] = label_array[id]
        """
        while(count != batch_size):
            id = random.randint(0, SIZE)
            if(id not in il_list):
                tensor[count] = smp.id2tensor(str(id))
                target[count] = label_array[id]
                count += 1
        """
        conf = model.extract(tensor)

        loss = 0.0
        for i in range(batch_size):
            loss += classfi_loss(conf[i], target[i], use_proposed=True)

        loss /= batch_size
        loss.backward()
        optimizer.step()

        loss_progress.append(loss.cpu().detach().numpy())

        if((epoch+1)%100 == 0):
            print(np.mean(loss_progress))

        print(epoch, loss.cpu().detach().numpy())
