import numpy as np
import math
import image_sampling as sampler
import stylenet
import load_model   # temporal import
from torch import Tensor
from math import log,pi
from collections import OrderedDict

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

tag_dict = OrderedDict()
with open("./labels.txt") as f:
    lines = f.readlines()
    for idx,e in enumerate(lines):
        e = e.strip()
        tag_dict[idx] = e

def gaussian_KL(p,q):
    """
    input: p,q (which includes Gaussian parameters(μ, Σ))
    output: KL distance between 2 input distributions
    """
    mp, cp = p
    mq, cq = q
    D = mp.shape[0]

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
        for j in range(10):
            tensors = sampler.tag_sampling(80, i)
            features = np.append(features, model.forward(tensors.cuda()).cpu().detach().numpy(), axis = 0)
        print(features.shape)

    colors = ["r","g","b"]
    label = [colors[int(x/2400)] for x in range(7200)]

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features)
    plt.scatter(x_pca[:,0], x_pca[:,1],c = label, alpha = 0.3, marker = ".")
    plt.legend(itemlist)
    plt.show()

def compute_gaussian(tag_idx):
    """
    DEPRECATED
    """
    features = np.empty([0, 128])
    model = load_model.model
    #model = stylenet.modelA
    model = model.cuda()
    model.train(False)

    for i in tag_idx:
        for j in range(3):
            tensors = sampler.tag_sampling(80, tag_idx)
            features = np.append(features, model.forward(tensors.cuda()).cpu().detach().numpy(), axis = 0)
        #print(features.shape)
    """
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(features)
    """
    f_list = []
    m_list = []
    c_list = []
    for i in range(len(tag_idx)):
        f_list.append(features[i*80:(i+1)*80])
        m_list.append(np.average(features[i*80:(i+1)*80], axis = 0))
        c_list.append(np.cov(features[i*80:(i+1)*80].T)+ 0.00001*np.eye(128))

    return m_list, c_list

def compute_gaussian2(tag_idx):
    features = np.empty([0, 128])
    model = load_model.model
    model = model.cuda()
    model.train(False)

    for j in range(20):
        tensors = sampler.tag_sampling(80, tag_idx)
        features = np.append(features, model.forward(tensors.cuda()).cpu().detach().numpy(), axis = 0)

    #mean = np.average(features[i*80:(i+1)*80], axis = 0)
    cov = np.cov((features[i*80:(i+1)*80].T))+ 0.001*np.eye(128)

    return cov

def compute_entropy(Cov):
    #print("det|cov| is {}".format(np.linalg.det(Cov)))
    return 0.5*(log(np.linalg.det(Cov))+128*(log(2*pi)+1))

def tag_mean_cov(tag_idx):
    features = np.empty([0, 128])
    model = load_model.model
    model = model.cuda()
    model.train(False)

    indices = list(sampler.search_id_by_tag(tag_idx))
    #print(np.array(indices).shape)
    count = 0
    for i in indices:
        #if count%10000 == 0: print(count)
        count += 1
        try:
            pix = sampler.id_sampling(i)
            features = np.append(features, model.forward(pix.cuda()).cpu().detach().numpy(), axis = 0)
        except:
            print("PASS THE ERROR")
        if count == 1420: break
    cov = np.cov(features.T)+ 0.0001*np.eye(128)
    mean =  np.mean(features, axis = 0)

    return mean, cov

def compute_image_dist():
    features = np.empty([0, 128])
    model = load_model.model
    model = model.cuda()
    model.train(False)

    count = 0
    for i in range(407000):
        if count%10000 == 0: print(count)
        count += 1
        try:
            pix = sampler.id_sampling(i)
            features = np.append(features, model.forward(pix.cuda()).cpu().detach().numpy(), axis = 0)
        except:
            print("PASS THE ERROR")

    cov = np.cov(features.T)+ 0.00001*np.eye(128)
    mean =  np.mean(features, axis = 0)

    np.save("datadist_mean.npy", mean)
    np.save("datadist_cov.npy", cov)

    return mean, cov

def io_test2():
    #pca_visualization_2d()
    tag_list = [37,41,55]
    tag_name = ["dress","jeans","hat"]

    m_list, c_list = compute_gaussian(tag_list)

    for t,c in zip(tag_name, c_list):
        print("{}:{}".format(t, np.linalg.det(c)))
        print("{}:{}".format(t, compute_entropy(c)))

if __name__ == "__main__":

    p = compute_image_dist()
    print(p)

    for i in range(66):
        q = tag_mean_cov(i)

        #num = sampler.number_of_image(i)
        #detc = round(math.log10(np.linalg.det(c)),2)
        print("{}:\t{}".format(tag_dict[i], gaussian_KL(p, q)))
