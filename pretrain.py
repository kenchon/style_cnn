import torch
import torch.nn as nn

import random
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import mymodules.line_notify as ln
from collections import OrderedDict as od

import load_model
import compute_similarity as cs
import image_sampling as smp
import train as tr
import stylenet
import test

cur_path = os.getcwd()
N_tags = 66
il_list = cs.il_list        # ilegal photo ids
SIZE = cs.SIZE
w = cs.weights
label_array = np.load("./noisy.npy")     # load numpy based labels
use_proposed = True
batch_size = 50
sum_w = cs.sum_w
delt = 0

def classfi_loss(conf, target, use_proposed = True):
    loss_c = 0
    if(use_proposed):
        for t in range(N_tags):
            y_k = (target[t] - 0.5) * 2
            loss_c += w[t]*torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy
        return loss_c/sum_w

    else:
        for t in range(N_tags):
            y_k = (target[t] - 0.5) * 2
            loss_c += torch.log(1 + torch.exp(-int(y_k) * conf[t]))    # cross entropy
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
    f_loss = open('loss_progress_pretrain.txt', mode = 'a')
    for i in loss_prog:
        f_loss.write(str(i)+'\n')
    f_loss.close()

    f_acc = open('acc_progress_pretrain.txt', mode = 'a')
    f_acc.write(str(temp_score)+'\n')
    f_acc.close()

def get_loss_acc_fig(loss_prog, temp_score):
    save_loss(loss_prog, temp_score)
    path_to_fig = get_prog_image()
    return path_to_fig


def get_prog_image(add = "", split = 100):
    with open('loss_progress_pretrain.txt','r') as f:
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

    with open('acc_progress_pretrain.txt','r') as f:
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

class LinearUnit(nn.Module):
    def __init__(self):
        super(LinearUnit, self).__init__()
        self.linear2 = nn.Linear(128, N_tags)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.linear2(input)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    epochs = 1000000
    split = 1000
    loss_progress = []

    model = stylenet.Stylenet()
    #model.load_state_dict(torch.load('linear_weight_softmax.pth'))
    model.cuda()
    model.train()

    learning_rate = 1e-4
    #optimizer = torch.optim.Adadelta(model.parameters(), weight_decay = 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)

    for epoch in range(epochs):
        if (epoch%split == 0 and epoch != 0):

            model_path = 'linear_weight_softmax_{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), 'optim_softmax.pth')

            temp_score = test.test(model_path)
            #temp_score = 0.588
            path_to_fig = get_loss_acc_fig(loss_progress, temp_score)
            ln.send_image_(path_to_fig, 'loss progress')
            ln.notify('{} {}'.format(str(temp_score), model_path))
            loss_progress = []

            #if(loss_progress[])

        # image sampling
        tensor = torch.Tensor(batch_size, 3, 384, 256).cuda()

        optimizer.zero_grad()
        #tensor.requires_grad_(False)
        target = list(range(batch_size))         # class labels
        count = 0
        while(count != batch_size):
            id = random.randint(0, SIZE)
            if(id not in il_list):
                tensor[count] = smp.id2tensor(str(id))
                target[count] = label_array[id]
                count += 1

        conf = model.extract(tensor)

        loss = 0
        for i in range(batch_size):
            loss += classfi_loss(conf[i], target[i], use_proposed)

        loss /= batch_size
        loss.backward()
        optimizer.step()

        print(epoch, loss.cpu().detach().numpy())
        loss_progress.append(loss.cpu().detach().numpy())
