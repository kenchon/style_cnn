import torch
import torch.nn as nn

import random
import numpy as np
import os
import matplotlib.pyplot as plt
import mymodules.line_notify as ln
from collections import OrderedDict as od

import load_model
import compute_similarity as cs
import image_sampling as smp
import train as tr
import stylenet

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
            #temp_loss = -(int(target[t]))*torch.log(conf[t]) - (1 - int(target[t]))*torch.log(1 - conf[t])    # cross entropy
            temp_loss = -(int(target[t]))*torch.log(conf[t])
            temp_loss *= w[t]
            loss_c += temp_loss
        return loss_c/sum_w

    else:
        for t in range(N_tags):
            loss_c += -(target[n][t])*conf[t] + (1 - target[n][t])*(1 - conf[t])    # cross entropy
        S = len(np.where(target[n]== 1)[0])
        loss_c = loss_c/S
        return loss_c

def get_saved_dict(path_to_weight = "linear_weight.pth"):

    dic = torch.load(path_to_weight)
    new_dict = od()
    new_dict['linear2.weight'] = dic['linear1.weight']
    new_dict['linear2.bias'] = dic['linear1.bias']

    return new_dict

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
    #print(label_array.shape) # TEMP:
    epochs = 1000000
    split = 1000
    loss_progress = []

    model = stylenet.Stylenet()
    model.cuda()

    learning_rate = 0.01
    #optimizer = torch.optim.Adadelta(model.parameters(), weight_decay = 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)

    for epoch in range(epochs):
        if (epoch%split == 0 and epoch != 0):
            tr.save_loss(loss_progress, add = "_pretrain")
            loss_progress = []
            path_to_fig = tr.send_prog_image(add = "_pretrain")
            ln.send_image_(path_to_fig, 'loss progress')
            torch.save(model.state_dict(), 'linear_weight.pth')
            torch.save(optimizer.state_dict(), 'optim.pth')
            learning_rate *= 0.9
            optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 5e-4)

        # image sampling
        tensor = torch.Tensor(batch_size, 3, 384, 256).cuda()

        #tensor.requires_grad_(False)
        target = list(range(batch_size))         # class labels
        count = 0
        while(count != batch_size):
            id = random.randint(0, SIZE)
            if(id not in il_list):
                tensor[count] = smp.id2tensor(str(id))
                target[count] = label_array[id]
                count += 1

        conf, _ = model.forward(tensor)

        loss = 0
        for i in range(batch_size):
            loss += classfi_loss(conf[i], target[i], use_proposed)

        loss /= batch_size
        optimizer.zero_grad()
        optimizer.step()

        print(epoch, loss.cpu().detach().numpy())
        loss_progress.append(loss.cpu().detach().numpy())
