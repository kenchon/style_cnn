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

cur_path = os.getcwd()
N_tags = 66
il_list = cs.il_list        # ilegal photo ids
SIZE = cs.SIZE
w = cs.weights
label_array = np.load("./noisy.npy")     # load numpy based labels
use_proposed = True
batch_size = 48

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
        self.logsoftmax = nn.LogSoftmax(dim = 1)

    def forward(self, input):
        x = self.linear2(input)
        x = self.logsoftmax(x)
        return x

if __name__ == '__main__':
    #print(label_array.shape) # TEMP:
    epochs = 15000
    loss_history = []

    extractor = load_model.model
    extractor.train(False)

    model = LinearUnit()
    # weight_dict = get_saved_dict()
    # model.load_state_dict(weight_dict)

    learning_rate = 0.01
    optimizer = torch.optim.Adadelta(model.parameters(),lr=learning_rate)

    tensor = torch.Tensor(batch_size, 3, 384, 256)
    #tensor.requires_grad_(False)

    for epoch in range(epochs):

        # image sampling
        target = list(range(batch_size))         # class labels
        count = 0
        while(count != batch_size):
            id = random.randint(0, SIZE)
            #print(id) # TEMP:

            if(id not in il_list):
                tensor[count] = smp.id2tensor(str(id))
                #print(label_array[id]) # TEMP:
                #print(list(np.where(label_array[id] == 1))[0]) # TEMP:
                target[count] = list(np.where(label_array[id] == 1))[0]
                count += 1

        #print(target) # TEMP:

        x_128dim = extractor.forward(tensor)
        #f.requires_grad = True

        log_y = model.forward(x_128dim)

        loss = 0
        for i in range(batch_size):
            loss += classfi_loss(log_y[i], target[i], use_proposed)
        #print(loss.shape)

        loss /= batch_size
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        loss = loss.cpu().detach().numpy()
        print(epoch, loss)
        loss_history.append(loss)

        if epoch%100 == 0 and epoch != 0:
            Y = np.array(loss_history)
            plt.plot(Y)
            path_to_img = cur_path + 'loss_progress.png'
            plt.savefig(path_to_img)
            ln.send_image_(path_to_img, 'loss progress')
            torch.save(model.state_dict(), 'linear_weight.pth')
            torch.save(optimizer.state_dict(), 'optim.pth')
