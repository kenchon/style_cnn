import numpy as np
import math
import image_sampling as sampler
import stylenet
import load_model   # temporal import
from torch import Tensor


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
    #model = model.cuda()
    model.train(False)
    size = 10

    for i in range(size):
        tensors = sampler.tag_sampling(1, 1)
        features.append(model.forward(Tensor(tensors)).detach().numpy())

    print(features)
