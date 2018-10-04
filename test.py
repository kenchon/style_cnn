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

def test(path_to_weight,do_classification, model,use_standard_output = True, seek_best = True):
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
    # TODO: need to modify in onder to process on/off mecanism
    """
    if do_classification:
        model = stylenet.Stylenet()
        model.load_state_dict(torch.load(path_to_weight))
        extract = model.extract
    else:
        model = load_model.model
        model.load_state_dict(torch.load(path_to_weight))
        extract = model.forward
    """

    """
    if do_classification:
        model = stylenet.Stylenet()
    else:
        model = load_model.model
    model.load_state_dict(torch.load(path_to_weight))
    model = model.cuda()
    """
    model = model
    model.eval()

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
    	                            param_grid=param_grid, cv=StratifiedKFold(y=Y, n_folds = 5))
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
    if use_standard_output: print("max:{} mean{}".format(max(scoresN), np.mean(scoresN)))
    # ln.notify(message)


    return np.mean(scoresN)


def matrix(path_to_weight,do_classification, use_standard_output = True, seek_best = True):
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
    # TODO: need to modify in onder to process on/off mecanism
    """
    if do_classification:
        model = stylenet.Stylenet()
        model.load_state_dict(torch.load(path_to_weight))
        extract = model.extract
    else:
        model = load_model.model
        model.load_state_dict(torch.load(path_to_weight))
        extract = model.forward
    """

    if do_classification:
        model = stylenet.Stylenet()
    else:
        model = load_model.model
    model.load_state_dict(torch.load(path_to_weight))
    model = model.cuda()

    model.eval()

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
    	                            param_grid=param_grid, cv=StratifiedKFold(y=Y, n_folds = 5))
    grid.fit(X, Y)

    # grid_scores_ contains parameter settings and scores
    score_dict = grid.grid_scores_
    # We extract just the scores
    scores = [x[1] for x in score_dict]

    clf = grid.best_estimator_
    clf.fit(X, Y)
    pre = clf.predict(X)

    N = 1
    scoresN = []
    trline = []
    preline = []

    i = 84
    rs=ShuffleSplit(n_splits=100, train_size=0.8,random_state=i)
    matrix = [[0]*5]*5
    for tr,ts in rs.split(X):
        clf.fit(X[tr],Y[tr])
        pre = clf.predict(X[ts])
        matrix += confusion_matrix(Y[ts], pre)
        trline.extend(Y[ts.tolist()])
        preline.extend(pre.tolist())
    print(matrix)
    print(classification_report(trline, preline,target_names = styles))

    """
    Acc, Recall, Precision computation
    """
    score = matrix
    sum = 0
    rig = 0

    acc_list = []
    pre_list = []
    rec_list = []

    for r in score:
        for e in r:
            sum += e
    count = 0
    now = 0
    for i in score:
        now =0
        for j in i:
            if now == count:
                rig += j
            now += 1
        count += 1

    print("Acc.\t", rig/sum)

    count = 0
    for r in score:
        sum_acc = 0

        now = 0
        for e in r:
            sum_acc += e
        for e in r:
            if count == now:
                acc_list.append(e/sum_acc)
            now += 1
        count += 1
    print("Rec.\t", np.mean(acc_list))

    score = list(np.array(score).T)


    pre_list = []

    count = 0
    for r in score:
        sum_pre = 0

        now = 0
        for e in r:
            sum_pre += e
        for e in r:
            if count == now:
                pre_list.append(e/sum_pre)
            now += 1
        count += 1
    print("Pre.\t", np.mean(pre_list))


    # message = "{} {} {}".format(max(scoresN),clf, path_to_weight)
    #if use_standard_output: print("max:{} mean{}".format(max(scoresN), np.mean(scoresN)))
    # ln.notify(message)


    #return np.mean(scoresN)


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
            feature = model.forward(tensor)
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

    N = 1
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
    #matrix('./result/params/prams_lr001_clas=True_epoch0_iter4000_7.pth', use_standard_output = True, do_classification = False)
    test("./result/params/prams_lr001_clas=False_pre_epoch0_iter12000.pth", use_standard_output = True, do_classification = False)
    #test('linear_weight_softmax.pth', do_classification = False)
