# coding: utf-8

import random
import numpy as np

def simlarity(img1, img2):
    union_score = 0
    sum_score = 0

    # 要素が１であるインデックスを抽出
    index1 = np.where(data[img1]==1)
    index2 = np.where(data[img2]==1)

    # img1とimg2に出現するアイテムのリストを作る
    img1_items = []
    img2_items = []
    comm_items = []

    item_count = 0
    for lis in metalist:
        if len(set(index1[0])&set(lis))!=0:
            img1_items.append(item_count)
        if len(set(index2[0])&set(lis))!=0:
            img2_items.append(item_count)
        if len(set(index1[0])&set(index2[0])&set(lis))!=0:
            comm_items.append(item_count)
        item_count+=1

    for i in comm_items:
        union_score += (weights[i] + 2)
    for i in set(img1_items)&set(img2_items)-set(comm_items):
        union_score += weights[i]
    for i in set(img1_items)|set(img2_items):
        sum_score += (weights[i] + 2)

    return union_score/sum_score

if __name__ == "__main__":
    """
    f1 = open("feat/colours.txt")
    f2 = open("feat/single_unique.txt")

    count = 0
    sublist = []
    metalist = []

    line1 = f1.readlines()
    line2 = f2.readlines()

    for i in line2:
        for j in line1:
            if i.strip() in j.strip():
                sublist.append(count)
            count += 1

        count = 0
        metalist.append(sublist)
        sublist = []
    """

    f_wei = open("weights.txt","r")

    weights = []
    values = [1,2,3,4,5]
    modifed = [0,2,4,8,10]

    for line in f_wei:
        line = line.strip("\n")
        line = line.split("\t")
        weights.append(line[1])

    data = np.load("./noisy.npy")

    SIZE = 407772
    PAIRS = 100000000000000000

    # compute similarity of images and save image triplet
    for i in range(PAIRS):
        # random sampling
        id1 = random.randint(0, SIZE)
        id2 = random.randint(0, SIZE)
        id3 = random.randint(0, SIZE)

        sim = simlarity(id1,id2)

        if id1 != id2 and sim > 0.75:
            sim_neg = 0
            while(sim_neg > 0.05):
                id3 = random.randint(0, SIZE)
                sim_neg = simlarity(id1,id3)

            s = str(id1)+","+str(id2)+","+str(id3)+","+str(round(sim,5))+","+str(round(sim_neg,5))
            #print(s)
            f_ids_and_sims = open("triplet_v3.csv","a")
            f_ids_and_sims.write(s+"\n")
            f_ids_and_sims.close()
