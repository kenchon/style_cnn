import warnings
warnings.filterwarnings('ignore')

import torch

from torch.autograd import Variable
from torch import autograd
from torchvision import models, transforms
from PIL import Image

from sklearn.metrics import confusion_matrix

import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

import pylab as pl

from sklearn.svm import SVC
#from sklearn.preprocessing import Scaler
from sklearn.datasets import load_iris
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

import subprocess as sp
import load_model
import image_sampling as sampler
import stylenet
import pandas as pd

import scipy.stats as sp
import mymodules.line_notify as ln

def test(path_to_weight, use_standard_output = False, seek_best = True, do_classification = True):
    """
    load hipster wars dataset
    """
    use_standard_output = True

    feature_list = []           # which keeps output of 128 dim features
    target_list = []            # which keeps corresponding target label

    home_dir = "../hipsterwars"
    styles = ["Hipster","Goth","Preppy","Pinup","Bohemian"]

    f_lables = open(home_dir+"/skills.txt")
    lines = f_lables.readlines()

    delta = 0.5

    img_paths = {}

    for style in styles:
        paths = []
        for line in lines:
            line = line.split(",")
            if line[0] == style:
                paths.append(line[1])
        img_paths[style] = paths[:int(len(paths)*delta)]

    """
    load model
    """
    # load model
    #model = stylenet.get_model()
    #model.load_state_dict(torch.load(path_to_weight))
    model = load_model.model
    model.load_state_dict(torch.load(path_to_weight))
    model.cuda()
    model.eval()        # use model as feature extractor

    """
    feature extraction
    """
    count = 0
    for style in styles:
        images = img_paths[style]
        for image in images:
            tensor = torch.Tensor(1, 3, 384, 256)
            path = "../hipsterwars/classes/"+ style + "/"+image+".jpg"
            tensor[0] = sampler.pix2tensor(sampler.id2pix(path, use_path = True))
            tensor = tensor.cuda()
            feature = model.forward(tensor)
            feature = feature.cpu()
            feature_list.append(feature.data.numpy())
            target_list.append(count)
        count += 1

    feature = np.array(feature_list)
    feature = feature[:,0,:]          # shape of (1893,1,128) to (1893,128)
    target = np.array(target_list)

    """
    tune the hyperparameter C
    """

    X = feature
    Y = target

    C_range = np.linspace(0.001, 0.05, 100)
    param_grid = dict(C=C_range)

    grid = GridSearchCV(svm.LinearSVC(penalty="l2", loss="squared_hinge", tol=1e-4, class_weight="balanced"),
    	                            param_grid=param_grid, cv=StratifiedKFold(y=Y, n_folds = 10))
    grid.fit(X, Y)

    # grid_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_
    # We extract just the scores
    scores = [x[1] for x in score_dict]

    clf = grid.best_estimator_
    clf.fit(X, Y)
    pre = clf.predict(X)

    N = 10
    scoresN = list(range(N))

    for i in range(N):
        clf = grid.best_estimator_
        rs=ShuffleSplit(n_splits=100, train_size=0.9,random_state=i)
        scores = cross_val_score(clf, feature, target, cv=rs)
        scoresN[i] = np.mean(scores)
        if use_standard_output: print(i, np.mean(scores))

    # message = "{} {} {}".format(max(scoresN),clf, path_to_weight)
    if use_standard_output: print(max(scoresN))
    # ln.notify(message)

    return max(scoresN)


def test2(parameter):
    """
    load hipster wars dataset
    """

    feature_list = []           # which keeps output of 128 dim features
    target_list = []            # which keeps corresponding target label

    home_dir = "../hipsterwars"
    styles = ["Hipster","Goth","Preppy","Pinup","Bohemian"]

    feature_list = []
    target_list = []

    f_lables = open(home_dir+"/skills.txt")
    lines = f_lables.readlines()

    delta = 0.5

    img_paths = {}

    for style in styles:
        paths = []
        for line in lines:
            line = line.split(",")
            if line[0] == style:
                paths.append(line[1])
        img_paths[style] = paths[:int(len(paths)*delta)]

    """
    load model
    """
    dictlist = []
    models  = parameter

    model_path = parameter
    best_dict = {}

    #model = load_model.model
    model = stylenet.Stylenet()
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()        # use network as feature extractor
    #print("loaded the model...")
    """
    feature extraction
    """
    count = 0
    for style in styles:
        images = img_paths[style]
        for image in images:
            tensor = torch.Tensor(1, 3, 384, 256)
            img_path = "../hipsterwars/classes/"+ style + "/"+image+".jpg"
            tensor[0] = sampler.pix2tensor(sampler.id2pix(img_path, use_path = True))
            tensor = tensor.cuda()
            feature = model.extract(tensor)
            feature = feature.cpu()
            feature_list.append(feature.data.numpy())
            target_list.append(count)
        count += 1

    feature = np.array(feature_list)
    feature = feature[:,0,:]          # shape of (1893,1,128) to (1893,128)
    target = np.array(target_list)
    print("got features...")

    """
    tune the hyperparameter C
    """

    #clf_sets = [(svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-3),
    #np.logspace(-1, 2, 50), feature, target)]

    X = feature
    #X = sp.stats.zscore(feature, axis = 1)
    Y = target

    C_range = np.linspace(0.001, 0.05, 100)
    gamma_range = np.linspace(0.0001, 0.0005, 30)
    #param_grid = dict(gamma=gamma_range, C=C_range)
    param_grid = dict(C=C_range)

    #grid = GridSearchCV(svm.SVC(kernel='rbf',tol=1e-4), param_grid=param_grid, cv=StratifiedKFold(y=Y))
    grid = GridSearchCV(svm.LinearSVC(penalty="l2", loss="squared_hinge", tol=1e-4, class_weight="balanced"),
    	                            param_grid=param_grid, cv=StratifiedKFold(y=Y, n_folds = 10))
    grid.fit(X, Y)

    #print("The best classifier is: ", grid.best_estimator_)

    # plot the scores of the grid
    # grid_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_

    # We extract just the scores
    scores = [x[1] for x in score_dict]

    clf = grid.best_estimator_
    clf.fit(X, Y)
    pre = clf.predict(X)

    best_dict["clf"] = clf

    #print(confusion_matrix(target, pre))
    #print(classification_report(target, pre,target_names = styles))


    print("max score on the tuning is {}".format(max(scores)))

    N = 100
    scoresN = np.zeros(N)

    for i in range(N):
        clf = grid.best_estimator_
        rs=ShuffleSplit(n_splits=100, train_size=0.9,random_state=i)
        scores = cross_val_score(clf, feature, target, cv=rs)
        scoresN[i] = np.mean(scores)
        print(i, np.mean(scores))

    # message = "{} {} {}".format(max(scoresN),clf, parameter)
    # ln.notify(message)

    dictlist.append(best_dict)
    print(best_dict)

    return str(max(scoresN))

if __name__ == "__main__":
    #test("./result/params/prams_lr001_clas=True_epoch0_iter100_5.pth")
    test('./result/params/prams_lr001_clas=True_epoch0_iter1000_5.pth', do_classification = False)
