from torch import Tensor
from torchvision import models, transforms
import torch
import csv
from PIL import Image
import random
import numpy as np
import numpy


label_array = np.load("./noisy.npy")     # load numpy based labels

with open("./photos.txt","r") as f:
    photos = f.readlines()
    N_img = len(photos)

def pix2tensor(pixel_img):
    u""" pix > tensor
    input: pixel based image
    output: image tensor
    """

    normalize = transforms.Normalize(
        mean=[0.5657177752729754, 0.5381838567195789, 0.4972228365504561],
        std=[0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    )
    preprocess = transforms.Compose([
        transforms.Resize((384,256)),
        transforms.ToTensor(),
        normalize
    ])

    return preprocess(pixel_img)

def id2pix(img_id):
    return Image.open(photos[int(photos[img_id][0])].strip())

def single_sampling(batch_size):
    u""" sampling method for classification
    input: batch_size
    output: tensor of (batch_size, 3, 384, 256)
    """

    label_array = np.load("./noisy.npy")     # load numpy based labels
    batch_tensors = torch.Tensor(batch_size, 3, 384, 256)
    labels = np.array(0)

    count = 0
<<<<<<< HEAD
    while count != (batch_size):
    #    try:
        idx = random.randint(0, N_img)
        batch_tensors[count] = pix2tensor(id2pix(photos[idx].strip()))
        print(photos[idx])
        labels.append(label_array(idx))
        count += 1
        """
        except:
            batch_tensors[count] = random.randint(0, N_img)
            labels = numpy.delete(labels, len(labels), axis = 0)    # delete last row if error occured
        """
=======

    while count != (batch_size):
        try:
            idx = random.randint(0, N_img)
            batch_tensors[count] = pix2tensor(id2pix(photos[idx]))
            labels.append(label_array(idx))
            count += 1
        except:
            batch[count] = random.randint(0, N_img)
            labels = numpy.delete(labels, len(labels), axis = 0)    # delete last row if error occured

>>>>>>> 063042ff1f53abd79a0c4e27321d087ecc71607c
    return batch_tensors, labels

def triplet_sampling(row, batch):
    u""" sampling method for triplet learning
    input: row, batch_size
    output: three tensors of (batch_size, 3, 384, 256) and similarity scores
    """

    ref_tensor = pos_tensor = neg_tensor = torch.Tensor(len(batch),3,384,256)
    batch_size = len(batch)

    sim_pos = list(range(batch_size))
    sim_neg = list(range(batch_size))

    count = 0
    bc = 0

    while count != (batch_size):
        try:
            idx = batch[count]
            ref_tensor[count] = pix2tensor(id2pix(row[idx][0]))
            pos_tensor[count] = pix2tensor(id2pix(row[idx][1]))
            neg_tensor[count] = pix2tensor(id2pix(row[idx][2]))
            sim_pos[count] = (float(row[idx][3]))
            sim_neg[count] = (float(row[idx][4]))
            count += 1
        except:
            batch[count] = random.randint(0,len(row))

    img = [ref_tensor,pos_tensor,neg_tensor]
    sim = [sim_pos,sim_neg]

    return img, sim

if __name__ == "__main__":
    print(single_sampling(24)[1])

