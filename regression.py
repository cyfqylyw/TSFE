import numpy as np
import matplotlib.pyplot as plt
import torch
import vit_pytorch
import torchvision
import torch
import pandas as pd
from PIL import Image, ImageFile
from modules.encoder import get_network  # encoder
from modules.TSFE_model import TSFE_model
from modules.dataset_loader import colorspaces, ResizeCrop
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split


def srocc_score(lst1, lst2):
    lst1_sort, lst2_sort = list(lst1).copy(), list(lst2).copy()
    N = len(lst1)
    lst1_sort.sort()
    lst2_sort.sort()

    sum = 0
    for i in range(N):
        vi = lst1_sort.index(lst1[i])
        pi = lst2_sort.index(lst2[i])
        sum += (vi - pi) ** 2

    return 1 - (6 * sum / (N * (N**2 - 1)))


def plcc_score(lst1, lst2):
    sum1, sum2, sum3 = 0, 0, 0
    m1, m2 = np.mean(lst1), np.mean(lst2)
    for i in range(len(lst1)):
        sum1 += (lst1[i] - m1) * (lst2[i] - m2)
        sum2 += (lst1[i] - m1) ** 2
        sum3 += (lst2[i] - m2) ** 2
    
    return sum1 / (np.sqrt(sum2 * sum3))

  
if __name__ == "__main__":
    mos = np.load('./data/mos.npy')
    feat_cl = np.load('./data/feat_cl(VGG16).npy')  # res50, res101, res152, VGG16, VGG16bn, googlenet, (VGG16)_ImageNet
    feat_vit = np.load('./data/feat_vit(ssim).npy')  # ssim, fsim, gmsd, sf, sg, fg, sfg, sfg2
    feat_all = np.concatenate((feat_cl, feat_vit), 1)
    seed = 42
    
    X_train, X_test, y_train, y_test = train_test_split(feat_all, mos, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/8, random_state=seed)
    
    max_val_srocc = -1
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ,1]:
        clf = Ridge(alpha=alpha)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        srocc_val = srocc_score(y_val_pred, y_val)
        if srocc_val > max_val_srocc:
            max_val_srocc = srocc_val
            best_alpha = alpha
            
            y_test_pred = clf.predict(X_test)
            cur_test_srocc = srocc_score(y_test_pred, y_test)
            cur_test_plcc = plcc_score(y_test_pred, y_test)
    
    print(cur_test_srocc, cur_test_plcc)
