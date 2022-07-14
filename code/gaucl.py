#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:05:30 2022

@author: gabri
"""

import numpy
import scipy.linalg
import scipy.special

def row(vector):
    return vector.reshape(1, vector.size)

def column(v):
    return v.reshape((v.size), 1)

class GaussianClassifier:
    def __init__(self, mode, tiedness, class_priors=None):
        if class_priors is None:
            class_priors = [0.5, 0.5]
        self.mode        = mode
        self.tiedness    = tiedness
    
    def train(self, DTR, LTR):
        D0 = DTR[:, LTR==0]
        D1 = DTR[:, LTR==1]
        
        mu0 = column(D0.mean(1))
        mu1 = column(D1.mean(1))
        
        DC0 = D0 - mu0
        DC1 = D1 - mu1
        Nc0 = D0.shape[1]
        Nc1 = D1.shape[1]
        
        C0         = numpy.dot(DC0, DC0.T) / Nc0
        C1         = numpy.dot(DC1, DC1.T) / Nc1
        naiveC0    = C0 * numpy.eye(C0.shape[0])
        naiveC1    = C1 * numpy.eye(C1.shape[0])
        tiedC      = (C0*Nc0 + C1*Nc1) / float(DTR.shape[1])
        tiedNaiveC = (naiveC0*Nc0 + naiveC1*Nc1) / float(DTR.shape[1])
        
        self.DTR, self.LTR         = DTR, LTR
        self.mu0, self.mu1         = mu0, mu1
        self.C0, self.C1           = C0, C1
        self.naiveC0, self.naiveC1 = naiveC0, naiveC1
        self.tiedC                 = tiedC
        self.tiedNaiveC            = tiedNaiveC
    
    def _logpdf_GAU_ND(self, X, mu, C):
        precision = numpy.linalg.inv(C)
        dimensions = X.shape[0]
        const = -0.5 * dimensions * numpy.log(2*numpy.pi)
        const += -0.5 * numpy.linalg.slogdet(C)[1]
    
        Y = []
        for i in range(X.shape[1]):
            x = X[:, i:i+1]
            res = const - 0.5 * numpy.dot((x-mu).T, numpy.dot(precision, (x-mu)))
            Y.append(res)
    
        return numpy.array(Y).ravel()
    
    def compute_lls(self, DTE):
        mu0, mu1    = self.mu0, self.mu1
        mode        = self.mode
        tiedness    = self.tiedness
        
        if mode == "full covariance" and tiedness == "untied":
            C0, C1 = self.C0, self.C1
        elif mode == "naive bayes" and tiedness == "untied":
            C0, C1 = self.naiveC0, self.naiveC1
        elif mode == "full covariance" and tiedness == "tied":
            C0, C1 = self.tiedC, self.tiedC
        elif mode == "naive bayes" and tiedness == "tied":
            C0, C1 = self.tiedNaiveC, self.tiedNaiveC
        else:
            print("ERROR: invalid mode in the gaussian classifier")
            quit()
            
        log_densities0 = self._logpdf_GAU_ND(DTE, mu0, C0)
        log_densities1 = self._logpdf_GAU_ND(DTE, mu1, C1)
        return log_densities0, log_densities1
    
    def compute_scores(self, DTE):
        log_densities0, log_densities1 = self.compute_lls(DTE)
        return log_densities1 - log_densities0
        
    #not necessary for the report
    def validate(self, DTE, LTE):
        predictedLabels = self.classify(DTE)
        wrongPredictions = (LTE != predictedLabels).sum()
        samplesNumber = DTE.shape[1]
        errorRate = float(wrongPredictions / samplesNumber * 100)
        return wrongPredictions, errorRate
    
    #not necessary for the report
    def classify(self, DTE):
        class_priors = self.class_priors
        
        log_densities0, log_densities1 = self.compute_lls(DTE)
        log_joint_densities0 = log_densities0 + numpy.log(class_priors[0])
        log_joint_densities1 = log_densities1 + numpy.log(class_priors[1])
        
        logScores = numpy.zeros((2, DTE.shape[1]))
        logScores[0, :] = log_joint_densities0
        logScores[1, :] = log_joint_densities1
    
        logMarginal = scipy.special.logsumexp(logScores, axis=0)
        logScores = logScores - row(logMarginal)
        scores = numpy.exp(logScores)
        predictedLabels = scores.argmax(axis=0)
        return predictedLabels
