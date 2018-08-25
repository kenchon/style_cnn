# coding: utf-8
import random
import numpy as np

f_wei = open("weights.txt","r")
weights = []

for line in f_wei:
    line = line.strip("\n")
    line = line.split("\t")
    weights.append(int(line[1]))

label_array = np.load("./noisy.npy")

def compute_simlarity(img1, img2):
    union_score = 0
    sum_score = 0
    item_size = label_array.shape[1]

    # 要素が１であるインデックスを抽出
    index1 = np.where(label_array[img1]==1)[0]
    index2 = np.where(label_array[img2]==1)[0]

    common = set(index1)&set(index2)
    one_side = (set(index1)|set(index2)) - common

    inter = 0
    union = 0

    for i in common:
        union += weights[i]
        inter += weights[i]

    for i in one_side:
        union += weights[i]

    #print("{}, {}".format(inter, union))
    if union == 0:
        return -1

    return inter/union

def sampling():
    SIZE = 407771

    # compute similarity of images and save image triplet
    while(True):
        # random sampling
        id1 = random.randint(0, SIZE)
        id2 = random.randint(0, SIZE)
        id3 = random.randint(0, SIZE)

        sim = compute_simlarity(id1,id2)

        if id1 != id2 and sim > 0.75:
            sim_neg = 0
            while(sim_neg > 0.05 or sim_neg < 0):
                id3 = random.randint(0, SIZE)
                sim_neg = compute_simlarity(id1,id3)

            s = str(id1)+","+str(id2)+","+str(id3)+","+str(round(sim,5))+","+str(round(sim_neg,5))
            f_ids_and_sims = open("triplet.csv","a")
            f_ids_and_sims.write(s+"\n")
            f_ids_and_sims.close()

if __name__ == "__main__":
    sampling()
