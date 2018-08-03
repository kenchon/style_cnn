import numpy as np
import math
import image_sampling as sampler
import stylenet
import load_model   # temporal import
from torch import Tensor

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def gaussian_KL(p,q):
    """
    input: p,q includes Gaussian parameters(μ, Σ)
    output: KL distance between 2 input distributions
    """
    mp, cp = p
    mq, cq = q
    D = mp.shape[0]
    #D = 2

    cp_inv = np.linalg.inv(cp)
    cq_det = np.linalg.det(cq)
    cp_det = np.linalg.det(cp)

    kl = 0.5 * (np.trace(np.dot(np.dot(mp-mq, (mp-mq).T) + cq, cp_inv)) + math.log(cp_det/cq_det) - D)

    return kl

def io_test():
    # i/o test
    p = [np.array([1, 1]),np.array([[1, 0],[0, 0.5]])]
    q = [np.array([1, 1]),np.array([[1, 0],[0, 1]])]

    print(gaussian_KL(p,q))

if __name__ == "__main__":
    features = []
    model = load_model.model
    #model = stylenet.modelA
    model = model.cuda()
    model.train(False)
    """
    tensors = sampler.tag_sampling(50, 1)
    print(tensors.shape)
    features.append(model.forward(tensors.cuda()).cpu().detach().numpy())
    features = np.array(features)[0]
    print(features.shape)

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features)
    plt.scatter(x_pca[:,0],x_pca[:,1])
    plt.show()
    """

    plotlist = [0, 2]
    itemlist = ["black","white"]

    for i in plotlist:
        print(i)
        c = 0
        features = []
        for j in range(10):
            tensors = sampler.tag_sampling(80, i)
            features.append(model.forward(tensors.cuda()).cpu().detach().numpy())

        features = np.array(features)[0]
        pca = PCA(n_components=2)
        x_pca = pca.fit_transform(features)
        plt.scatter(x_pca[:,0], x_pca[:,1])
        plt.legend(itemlist)
        c+=1
    plt.show()
