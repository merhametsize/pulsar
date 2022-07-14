#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:06:11 2022

@author: gabri
"""

import dataset
import gaucl
import logreg
import svm
import gmm
from xvalidator import compute_min_DCF

def main():
    DTR, LTR = dataset.load_data("PULSAR_TRAIN.txt")
    DTE, LTE = dataset.load_data("PULSAR_TEST.txt")
    
    DTR, mu, sigma = dataset.normalize_zscore(DTR)
    DTE, mu, sigma = dataset.normalize_zscore(DTE, mu, sigma)
    
    print("MVG")
    for pi in [0.5, 0.1, 0.9]:
        gc = gaucl.GaussianClassifier("full covariance", "tied")
        gc.train(DTR, LTR)
        scores = gc.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")
    
    print("LR")
    for pi in [0.5, 0.1, 0.9]:
        lr = logreg.LogRegClassifier(0, 0.5)
        lr.train(DTR, LTR)
        scores = lr.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")
    
    print("SVM")
    for pi in [0.5, 0.1, 0.9]:
        s = svm.SupportVectorMachines(1, "linear", 0.5)
        s.train(DTR, LTR)
        scores = s.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")
    
    print("GMM")
    for pi in [0.5, 0.1, 0.9]:
        g = gmm.GMM_classifier(3, "full", "untied")
        g.train(DTR, LTR)
        scores = g.compute_scores(DTE)
        min_DCF = compute_min_DCF(scores, LTE, pi, 1, 1)
        print(min_DCF)
    print("")


main()