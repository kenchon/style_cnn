import image_sampling as sampler
import subprocess as sp
import load_model
import numpy as np

DATASET_SIZE = 407772
with open("./photos.txt","r") as f:
    photos = f.readlines()

"""
This module removes the ilegal-format image (whose pixel is NOT represented as
[3, 384, 256] i.e. monochrome images) or something.
"""

def remover(img_path):
    #sp.getoutput("rm {}".format(img_path))
    print("rm {}".format(img_path))

def rm_operator(delete_list):
    for i in delete_list:
        path_to_img = sampler.id2path(i)
        remover(path_to_img)

def checker():
    model = load_model.model
    model = model.cuda()
    model.train(False)

    ilegal_list = []

    count = 0
    for i in range(130000, DATASET_SIZE):
        if i%10000 == 0: print(i)
        try:
            pix = sampler.id_sampling(i)
            model.forward(pix.cuda()).cpu().detach().numpy()
        except:
            ilegal_list.append(i)
            print("caught:\t"+str(i))
    return ilegal_list

if __name__ == "__main__":
    il_list = [137718, 137719, 159150, 159154, 159155, 159157, 159158, 159159, 159160, 172834, 172836, 172838, 172841, 175080, 200126, 217097, 218044, 223835, 296205, 300948, 337813, 374843, 204221, 204222, 204223, 275625]
    #rm_operator(il_list)

    for i in il_list:
        photos[i] = "LINE_TO_BE_DELETED"

    new_photos = [x for x in photos if x!= "LINE_TO_BE_DELETED"]
    print(len(new_photos))

    with open("./photos_.txt", mode='w') as f:
        f.writelines(new_photos)
