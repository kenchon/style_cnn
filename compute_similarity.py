# coding: utf-8
import random
import numpy as np

use_proposed = False

f_wei = open("weights.txt","r")
weights = []
il_list = [137718, 137719, 159150, 159154, 159155, 159157, 159158, 159159, 159160, 172834, 172836, 172838, 172841, 175080, 200126, 217097, 218044, 223835, 296205, 300948, 337813, 374843, 204221, 204222, 204223, 275625]

for line in f_wei:
    line = line.strip("\n")
    line = line.split("\t")
    weights.append(int(line[1]))

label_array = np.load("./noisy.npy")

def compute_simlarity(img1, img2, use_proposed = True):
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

    if(use_proposed):
        for i in common:
            union += weights[i]
            inter += weights[i]

        for i in one_side:
            union += weights[i]

    else:
        for i in common:
            union += 1
            inter += 1

        for i in one_side:
            union += 1

    #print("{}, {}".format(inter, union))
    if union == 0:
        return -1

    return inter/union

def sampling(use_proposed = True):
    SIZE = 405588

    # compute similarity of images and save image triplet
    while(True):
        # random sampling
        while(True):
            id1 = random.randint(0, SIZE)
            id2 = random.randint(0, SIZE)
            id3 = random.randint(0, SIZE)
            if(id1 not in il_list and id2 not in il_list and id3 not in il_list):
                break

        sim = compute_simlarity(id1, id2, use_proposed)

        if id1 != id2 and sim > 0.75:
            sim_neg = 0
            while(sim_neg > 0.05 or sim_neg < 0):
                id3 = random.randint(0, SIZE)
                sim_neg = compute_simlarity(id1, id3, use_proposed)

            s = str(id1)+","+str(id2)+","+str(id3)+","+str(round(sim,5))+","+str(round(sim_neg,5))
            f_ids_and_sims = open("triplet_pre.csv","a")
            f_ids_and_sims.write(s+"\n")
            f_ids_and_sims.close()

if __name__ == "__main__":
    sampling(use_proposed = False)
