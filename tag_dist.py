import numpy as np
import math
import image_sampling as sampler
import stylenet
import load_model   # temporal import
from torch import Tensor

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

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

def pca_visualization_3d():
    features = np.empty([0, 128])
    model = load_model.model
    #model = stylenet.modelA
    model = model.cuda()
    model.train(False)

    plotlist = [37,41,65]
    itemlist = ["dress","jeans","belt"]

    fig = plt.figure()
    ax = Axes3D(fig)

    for i in plotlist:
        for j in range(30):
            tensors = sampler.tag_sampling(80, i)
            features = np.append(features, model.forward(tensors.cuda()).cpu().detach().numpy(), axis = 0)
        print(features.shape)

    colors = ["r","g","b"]
    label = [colors[int(x/8000)] for x in range(24000)]

    features = np.array(features)
    pca = PCA(n_components=3)
    x_pca = pca.fit_transform(features)
    ax.scatter3D(x_pca[:,0], x_pca[:,1], x_pca[:,2],c = label)
    ax.legend(itemlist)

    plt.show()

def pca_visualization_2d():
    features = np.empty([0, 128])
    model = load_model.model
    #model = stylenet.modelA
    model = model.cuda()
    model.train(False)

    plotlist = [37,41,65]
    itemlist = ["dress","jeans","belt"]

    for i in plotlist:
        for j in range(30):
            tensors = sampler.tag_sampling(80, i)
            features = np.append(features, model.forward(tensors.cuda()).cpu().detach().numpy(), axis = 0)
        print(features.shape)

    colors = ["r","g","b"]
    label = [colors[int(x/2400)] for x in range(7200)]

    features = np.array(features)
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features)
    plt.scatter(x_pca[:,0], x_pca[:,1],c = label, alpha = 0.25)
    plt.legend(itemlist)
    plt.show()

if __name__ == "__main__":
    pca_visualization_2d()
