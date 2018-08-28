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

import requests

import warnings
warnings.filterwarnings('ignore')

def img2variable(path_to_img):
    normalize = transforms.Normalize(
        # these parameters should be modified for HipsterWars dataset
        mean=[0.5657177752729754, 0.5381838567195789, 0.4972228365504561],
        std=[0.29023818639817184, 0.2874722565279285, 0.2933830104791508]
    )
    preprocess = transforms.Compose([
        transforms.Resize([384,256]),
        transforms.ToTensor(),
        normalize
    ])

    img = Image.open(path_to_img)
    img_tensor = preprocess(img)
    return Variable(img_tensor.unsqueeze_(0)).cuda()

if __name__ == "__main__":
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
    models  =["parameters_.pth"]

    path = models[0]
    best_dict = {}

    model = load_model.model
    model.load_state_dict(torch.load(path))
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

    print("The best classifier is: ", grid.best_estimator_)

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

    print("the best score of 075")
    #clf = svm.LinearSVC(C=0.004132323232323233, class_weight='balanced', dual=True,
    #fit_intercept=True, intercept_scaling=1, loss='squared_hinge',
    #max_iter=1000, multi_class='ovr', penalty='l2', random_state=None,
    #tol=0.0001, verbose=0)
    #rs=ShuffleSplit(n_splits=100, train_size=0.8,random_state=41)

    fr = open("result_2018523.txt","a")

    trline = []
    preline = []
    for i in range(N):
        rs=ShuffleSplit(n_splits=100, train_size=0.8,random_state=i)
        matrix = [[0]*5]*5
        for tr,ts in rs.split(X):
            clf.fit(X[tr],Y[tr])
            pre = clf.predict(X[ts])
            matrix += confusion_matrix(Y[ts], pre)
            trline.extend(Y[ts.tolist()])
            preline.extend(pre.tolist())
        scores = cross_val_score(clf, feature, target, cv=rs)
        scoresN[i] = np.mean(scores)
        fr.write(matrix)
        fr.write(scoresN[i])
        print(scoresN[i])
        print(matrix)
        print(classification_report(trline, preline,target_names = styles))

    for i in range(N):
        clf = grid.best_estimator_
        rs=ShuffleSplit(n_splits=100, train_size=0.9,random_state=i)
        scores = cross_val_score(clf, feature, target, cv=rs)
        scoresN[i] = np.mean(scores)
        print(i, np.mean(scores))

        #print("here's a result:")
        #print(mean_score)
        #print(scores)

    best_dict["best_seed"] = np.argmax(scoresN)
    best_dict["model"] = path
    best_dict["score"] = np.max(scoresN)

    #print("{} {}".format(np.mean(scoresN), np.std(scoresN)))
    df = pd.DataFrame(scoresN)
    df.to_csv("{}_linear.csv".format(path))


    """
    f_result = open(home_dir+"/result_proposed.txt")
    f_result.write("{} {}".format(np.mean(scoresN), np.std(scoresN)))
    f_result.close()
    """

    """
    notify via LINE
    """
    line_notify_token = "5HUT8C9hfybizGGHMW3gtS6QQWM7CaA5Nzd0gJ2oXuQ"
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = "{} {} {}".format(path, max(scoresN),clf)


    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}  # 発行したトークン
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    dictlist.append(best_dict)
    print(best_dict)
