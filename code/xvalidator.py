#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:08:55 2022

@author: gabri
"""

import numpy
from dataset import PCA_reduce, normalize_zscore, gaussianize

class CrossValidator:
    def __init__(self, classifier, D, L):
        self.D = D
        self.L = L
        self.classifier = classifier
        
    def kfold(self, options):
        D = self.D
        L = self.L
        classifier = self.classifier
        
        K = options["K"]
        m = options["m"]
        gaussianization = options["gaussianization"]
        pi = options["pi"]
        (cfn, cfp) = options["costs"]
        
        samplesNumber = D.shape[1]
        N = int(samplesNumber / K)
        
        numpy.random.seed(seed=0)
        indexes = numpy.random.permutation(D.shape[1])
        
        scores = numpy.array([])
        labels = numpy.array([])
        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            
            idxTrainLeft = indexes[0:i*N]
            idxTrainRight = indexes[(i+1)*N:]
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]   
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            #zed-normalizes the data with the mu and sigma computed with DTR
            DTR, mu, sigma = normalize_zscore(DTR)
            DTE, mu, sigma = normalize_zscore(DTE, mu, sigma)
            
            if gaussianization == "yes":
                DTR, DTE = gaussianize(DTR, DTE)
            
            if m is not None: #PCA needed
                DTR, P = PCA_reduce(DTR, m)
                DTE = numpy.dot(P.T, DTE)
                
            classifier.train(DTR, LTR)
            scores_i = classifier.compute_scores(DTE)
            scores = numpy.append(scores, scores_i)
            labels = numpy.append(labels, LTE)
        min_DCF = compute_min_DCF(scores, labels, pi, cfn, cfp)
        return min_DCF, scores, labels
    
def assign_labels(scores, pi, cfn, cfp, threshold=None):
    if threshold is None:
        threshold = -numpy.log(pi*cfn) + numpy.log((1-pi)*cfp)
    
    predictions = scores > threshold
    return numpy.int32(predictions)

def compute_confusion_matrix(predictions, labels):
    C = numpy.zeros((2, 2))
    C[0, 0] = ((predictions == 0) * (labels == 0)).sum()
    C[0, 1] = ((predictions == 0) * (labels == 1)).sum()
    C[1, 0] = ((predictions == 1) * (labels == 0)).sum()
    C[1, 1] = ((predictions == 1) * (labels == 1)).sum()
    return C

def compute_emp_bayes(CM, pi, cfn, cfp):
    fnr = CM[0, 1] / (CM[0, 1] + CM[1, 1])
    fpr = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    emp_bayes_risk = pi*cfn*fnr + (1-pi)*cfp*fpr
    return emp_bayes_risk

def compute_normalized_emp_bayes(CM, pi, cfn, cfp):
    emp_bayes_risk = compute_emp_bayes(CM, pi, cfn, cfp)
    return emp_bayes_risk / min(pi*cfn, (1-pi)*cfp)

def compute_act_DCF(scores, labels, pi, cfn, cfp, threshold=None):
    predictions = assign_labels(scores, pi, cfn, cfp, threshold)
    CM = compute_confusion_matrix(predictions, labels)
    return compute_normalized_emp_bayes(CM, pi, cfn, cfp)

def compute_min_DCF(scores, labels, pi, cfn, cfp):
    thresholds = numpy.array(scores)
    thresholds.sort()
    
    minus_inf = numpy.array([-numpy.inf])
    plus_inf = numpy.array([numpy.inf])
    numpy.concatenate([minus_inf, thresholds.ravel(), plus_inf])
    dcf_list = []
    for t in thresholds:
        act_DCF = compute_act_DCF(scores, labels, pi, cfn, cfp, t)
        dcf_list.append(act_DCF)
    return numpy.array(dcf_list).min()

def plot_bayes_error(scores, labels, title):
    import pylab
    act_y = []
    min_y = []
    pis = numpy.linspace(-3, 3, 21)
    
    for p in pis:
        pi = 1.0 / (1.0 + numpy.exp(-p))
        min_y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        act_y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    pylab.figure()
    pylab.plot(pis, min_y, color="b")
    pylab.plot(pis, act_y, color="r")
    pylab.ylim(0, 1.1)
    pylab.savefig('%s.pdf' % title)

def plot_ROC(scores, labels):
    import pylab
    thresholds = numpy.array(scores)
    thresholds.sort()
    
    fpr = numpy.zeros(thresholds.size)
    tpr = numpy.zeros(thresholds.size)
    for i, t in enumerate(thresholds):
        predictions = numpy.int32(scores > t)
        CM = compute_confusion_matrix(predictions, labels)
        tpr[i] = CM[1, 1] / (CM[1, 1] + CM[0, 1])
        fpr[i] = CM[1, 0] / (CM[0, 0] + CM[1, 0])
    pylab.plot(fpr, tpr)
    pylab.show()