from torch import Tensor
from torchvision import models, transforms
import torch
import csv
from PIL import Image
import random
import numpy as np
import numpy
from math import *


#label_array = np.load("./verified.npy")     # load numpy based labels
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

def id2pix(img_id, use_path = False):
    if(use_path):
        return Image.open(img_id)
    else:
        #print(img_id, type(img_id))
        img_id = img_id.strip()
        img_id = int(img_id)
        return Image.open("../fashion550k/photos/"+photos[img_id].strip())

def id2tensor(id, use_path = False):
    return pix2tensor(id2pix(id, use_path))

def single_sampling(batch_size):
    u""" sampling method for classification
    input: batch_size
    output: tensor of (batch_size, 3, 384, 256)
    """
    batch_tensors = torch.Tensor(batch_size, 3, 384, 256)
    labels = []

    count = 0

    while count != (batch_size):
        """
        try:
            idx = random.randint(0, N_img)
            batch_tensors[count] = pix2tensor(id2pix(photos[idx]))
            labels.append(label_array(idx))
            count += 1
        except:
            idx = random.randint(0, N_img)
            labels = numpy.delete(labels, len(labels), axis = 0)    # delete last row if error occured
        """
        idx = random.randint(0, N_img)
        batch_tensors[count] = pix2tensor(id2pix(photos[idx]))
        labels.append(label_array[idx])
        count += 1
        print(count)
        #idx = random.randint(0, N_img)
        #labels = numpy.delete(labels, len(labels), axis = 0)    # delete last row if error occured

    return batch_tensors, labels

def tag_sampling(size, tag_idx):
    tensors = torch.Tensor(size, 3, 384, 256)
    count = 0
    photo_ids = np.where(label_array[:, tag_idx] == 1)[0]

    while count != (size):
        idx = random.randint(0, photo_ids.shape[0])     # randomly sample the photo id
        tensors[count] = pix2tensor(id2pix(photos[idx]))
        count += 1
    return tensors


def triplet_sampling(row, batch, do_classification = True):
    u""" sampling method for triplet learning
    input: row, batch
    output: three tensors of (batch_size, 3, 384, 256) and similarity scores
    """
    ref_tensor = pos_tensor = neg_tensor = torch.Tensor(len(batch),3,384,256)
    batch_size = len(batch)

    sim_pos = list(range(batch_size))
    sim_neg = list(range(batch_size))
    target = list(range(batch_size))
    target_npy = list(range(batch_size))

    count = 0
    bc = 0

    while count != (batch_size):
        idx = batch[count]
        ref_tensor[count] = pix2tensor(id2pix(row[idx][0]))
        pos_tensor[count] = pix2tensor(id2pix(row[idx][1]))
        neg_tensor[count] = pix2tensor(id2pix(row[idx][2]))
        sim_pos[count] = (float(row[idx][3]))
        sim_neg[count] = (float(row[idx][4]))
        target[count] = list(np.where(label_array[int(row[idx][2])] == 1))[0]
        target_npy[count] = label_array[int(row[idx][2])]
        count += 1

    img = [ref_tensor, pos_tensor, neg_tensor]
    sim = [sim_pos, sim_neg]

    if(do_classification):
        return img, sim, target, target_npy
    else:
        return img, sim



def id_sampling(id):
    tensor = torch.Tensor(1, 3, 384, 256)
    tensor[0] = pix2tensor(id2pix(photos[id]))
    return tensor

def id2path(img_id):
    return "../fashion550k/photos/"+photos[img_id].strip()

def search_id_by_tag(tag_idx):
    return np.where(label_array[:, tag_idx] == 1)[0]

def number_of_image(tag_idx):
    return np.where(label_array[:, tag_idx] == 1)[0].shape[0]

if __name__ == "__main__":
    tensor = id_sampling(1)
    print(number_of_image(1))
