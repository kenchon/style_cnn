import image_sampling as sampler
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import torch
from math import sqrt

SIZE = 405588
il_list = [137718, 137719, 159150, 159154, 159155, 159157, 159158, 159159, 159160, 172834, 172836, 172838, 172841, 175080, 200126, 217097, 218044, 223835, 296205, 300948, 337813, 374843, 204221, 204222, 204223, 275625]


idlist = set(list(range(SIZE))) - set(il_list)
idlist = list(idlist)

def main():
    count = 0
    temp = torch.zeros((384,256,3),dtype=torch.float64)
    temp.cuda()

    for id in idlist:
        img_np = torch.tensor(np.asarray(sampler.id2pix(str(id))),dtype=torch.float64)
        img_np.cuda()
        """
        for x in range(384):
            for y in range(256):
                for c in range(3):
                    temp[x][y][c] += img_np[x][y][c]/255
        """
        try:
            temp += img_np/255
        except:
            print("ERROR")
        count += 1
        if count % 10000 == 0: print(count)
    temp = temp.numpy()

    cp.save('pix_mean',temp)
    plt.imshow(temp)
    plt.show()


if __name__ == "__main__":
    """
    a = np.asarray(sampler.id2pix("1"))
    print(a.transpose(2,0,1).shape)
    plt.imshow(a)
    plt.show()
    """
    #main()
    arr = np.load("pix_mean.npy")
    im = arr*255/SIZE
    arr = arr/SIZE
    #print(arr)
    im = np.array(im, dtype = int)
    plt.imshow(im)
    plt.show()
    print(arr.shape)

    tensor = torch.tensor(arr)

    ch_mean = []
    ch_var = []
    for c in range(3):
        ch = tensor[:,:,c].view(384*256)
        ch_mean.append(np.mean(ch.numpy()))
        ch_var.append(sqrt(np.var(ch.numpy())))
    print(ch_mean)
    print(ch_var)
