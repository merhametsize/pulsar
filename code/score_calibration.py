#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:02:51 2022

@author: gabri
"""

import gaucl
import xvalidator
import dataset
import logreg
import svm
import gmm

K = 3

def row(vector):
    return vector.reshape(1, vector.size)

def calibrate_scores():
    D, L = dataset.load_data("PULSAR_TRAIN.txt")
    
    #MVG
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}
    gc = gaucl.GaussianClassifier("full covariance", "tied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, scores, labels = v.kfold(options)
    #xvalidator.plot_bayes_error(scores, labels, "MVG")
    
    lr = logreg.LogRegClassifier(0, 0.5)
    v = xvalidator.CrossValidator(lr, row(scores), labels)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "MVGcalibrated")
    
    #LR
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 0}
    lr = logreg.LogRegClassifier(options["l"], options["pT"])
    v = xvalidator.CrossValidator(lr, D, L)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "LR")
    
    lr = logreg.LogRegClassifier(0, 0.5)
    v = xvalidator.CrossValidator(lr, row(scores), labels)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "LRcalibrated")
    
    #SVM
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "linear",
               "C": 1}
    s = svm.SupportVectorMachines(options["C"], options["mode"], options["pT"])
    v = xvalidator.CrossValidator(s, D, L)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "SVM")
    
    lr = logreg.LogRegClassifier(0, 0.5)
    v = xvalidator.CrossValidator(lr, row(scores), labels)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "SVMcalibrated")
    
    #GMM
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "full",
               "tiedness": "untied",
               "n": 3} #number of doublings in LBG, #components=2^n
    g = gmm.GMM_classifier(options["n"], options["mode"], options["tiedness"])
    v = xvalidator.CrossValidator(g, D, L)
    min_DCF, scores, labels = v.kfold(options)
    xvalidator.plot_bayes_error(scores, labels, "GMM")

calibrate_scores()