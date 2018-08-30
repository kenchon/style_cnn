import image_sampling as sampler
import subprocess as sp
import load_model
import numpy as np

DATASET_SIZE = 407772

def remover(img_id):
    return 0

def checker():
    model = load_model.model
    model = model.cuda()
    model.train(False)

    ilegal_list = []

    count = 0
    for i in range(407772):
        if i%10000 == 0: print(i)
        try:
            pix = sampler.id_sampling(i)
            model.forward(pix.cuda()).cpu().detach().numpy()
        except:
            ilegal_list.append(i)
            print("caught:\t"+i)
    return ilegal_list

if __name__ == "__main__":
    print(checker())
