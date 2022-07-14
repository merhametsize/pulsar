#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 21:04:35 2022

@author: gabri
"""

import dataset
import gaucl
import xvalidator
import logreg
import svm
import gmm

import matplotlib.pyplot as plt
import numpy

K = 3 #must be consistent between all models

def main():
    D, L = dataset.load_data("PULSAR_TRAIN.txt")
    
    plot_figures(D, L)
    
    gaussian_classifiers(D, L)
    
    plot_lambda_minDCF(D, L)
    logistic_regression(D, L)
    
    plot_C_gamma_minDCF(D, L)
    SVM(D, L)
    
    GMM(D, L)
    
def gaussian_classifiers(D, L):
    options = {"m": None, #No PCA
               "gaussianization": "no",
               "K": K, 
               "pi": 0.5, 
               "costs": (1, 1)}
    
    options["gaussianization"] = "no"
    for options["m"] in [6, 5]:
        for options["pi"] in [0.5, 0.1, 0.9]:
            print(options)
            test_gauss_classifiers(D, L, options)
    
def logistic_regression(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 0}
    
    for options["gaussianization"] in ["yes"]:
        print("")
        for options["pi"] in [0.5, 0.1, 0.9]:
            print("")
            for options["pT"] in [0.5, 0.1, 0.9]:
                print(options)
                test_logistic_regression(D, L, options)

def SVM(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "RBF",
               "C": 1,
               "gamma": numpy.exp(-3)}
    
    
    for options["pi"] in [0.5, 0.1, 0.9]:
        print("")
        for options["pT"] in [0.5, 0.1, 0.9]:
            print(options)
            test_SVM(D, L, options)

def GMM(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "K": K,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "full",
               "tiedness": "untied",
               "n": 3} #number of doublings in LBG, #components=2^n
    for options["n"] in [2, 3]:
        for options["mode"] in ["full", "naive"]:
            for options["tiedness"] in ["untied", "tied"]:
                print(options)
                test_GMM(D, L, options)

def test_logistic_regression(D, L, options):
    l = options["l"]
    pT = options["pT"]
    lr = logreg.LogRegClassifier(l, pT)
    v = xvalidator.CrossValidator(lr, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Logistic regression: %.3f" % min_DCF)
    return min_DCF

def test_gauss_classifiers(D, L, options):    
    gc = gaucl.GaussianClassifier("full covariance", "untied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Full covariance - untied: %.3f" % min_DCF)
    
    gc = gaucl.GaussianClassifier("naive bayes", "untied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Naive bayes - untied: %.3f" % min_DCF)
    
    gc = gaucl.GaussianClassifier("full covariance", "tied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Full covariance - tied: %.3f" % min_DCF)
    
    gc = gaucl.GaussianClassifier("naive bayes", "tied")
    v = xvalidator.CrossValidator(gc, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("Naive Bayes - tied: %.3f" % min_DCF)
    print("")
    
def test_SVM(D, L, options):
    s = svm.SupportVectorMachines(options["C"], options["mode"], options["pT"], gamma=options["gamma"])
    v = xvalidator.CrossValidator(s, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("SVM: %.3f" % min_DCF)
    return min_DCF

def test_GMM(D, L, options):
    g = gmm.GMM_classifier(options["n"], options["mode"], options["tiedness"])
    v = xvalidator.CrossValidator(g, D, L)
    min_DCF, _, _ =v.kfold(options)
    print("GMM: %.3f" % min_DCF)
    return min_DCF

def plot_lambda_minDCF(D, L):
    options = {"m": None,
               "gaussianization": "yes",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "l": 0}
    
    pis = [0.5, 0.1, 0.9]
    lambdas = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["l"] in lambdas:
            print(options)
            min_DCF = test_logistic_regression(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(lambdas, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.savefig("lambda_minDCF_gau.pdf")

def plot_C_minDCF(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "RBF",
               "C": 1}
    
    pis = [0.5, 0.1, 0.9]
    C = [0.5e-4, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1, 1e-1, 0.5, 1, 0.5e1, 1e1, 0.5e2, 1e2]
    min_DCFs = {pi: [] for pi in pis}
    for options["pi"] in pis:
        print("")
        for options["C"] in C:
            print(options)
            min_DCF = test_SVM(D, L, options)
            min_DCFs[options["pi"]].append(min_DCF)
    plt.figure()
    for pi in pis:
        plt.plot(C, min_DCFs[pi], label='prior='+str(pi))
    plt.legend()
    plt.semilogx()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.savefig("C_minDCF.pdf")

def plot_C_gamma_minDCF(D, L):
    options = {"m": None,
               "gaussianization": "no",
               "K": K,
               "pT": 0.5,
               "pi": 0.5,
               "costs": (1, 1),
               "mode": "RBF",
               "C": 1,
               "gamma": numpy.exp(-1)}
    
    gammas = [numpy.exp(-1), numpy.exp(-2), numpy.exp(-3)]
    C = [0.5e-4, 1e-4, 0.5e-3, 1e-3, 0.5e-2, 1e-2, 0.5e-1, 1e-1, 0.5, 1, 0.5e1, 1e1, 0.5e2, 1e2]
    min_DCFs = {gamma: [] for gamma in gammas}
    for options["gamma"] in gammas:
        print("")
        for options["C"] in C:
            print(options)
            min_DCF = test_SVM(D, L, options)
            min_DCFs[options["gamma"]].append(min_DCF)
    plt.figure()
    for gamma in gammas:
        plt.plot(C, min_DCFs[gamma], label='γ='+str(gamma))
    plt.legend()
    plt.semilogx()
    plt.xlabel("C")
    plt.ylabel("minDCF")
    plt.savefig("C_gamma_minDCF.pdf")
    
def plot_figures(D, L):
    print("Plotting data... ", end=" ")
    
    D_gauss = dataset.gaussianize(D)
    D_norm, mu, sigma = dataset.normalize_zscore(D)
    
    dataset.plot_scatter(D_norm, L, "PCA  raw data")
    dataset.plot_scatter(D_gauss, L, "PCA - gaussianized")
    
    dataset.plot_histograms(D_norm, L, "raw")
    dataset.plot_histograms(D_gauss, L, "gaussianized")
    
    dataset.heatmap(D_norm, "raw", "Blues")
    dataset.heatmap(D_norm[:, L==1], "raw_pos", "Greens")
    dataset.heatmap(D_norm[:, L==0], "raw_neg", "Reds")
    
    dataset.heatmap(D_gauss, "gaussianized", "Blues")
    
    D7, _P = dataset.PCA_reduce(D, 7)
    dataset.heatmap(D7, "raw_m7", "Blues")
    dataset.heatmap(D7[:, L==1], "raw_pos7", "Greens")
    dataset.heatmap(D7[:, L==0], "raw_neg7", "Reds")
    
    D6, _P = dataset.PCA_reduce(D, 6)
    dataset.heatmap(D6, "raw_m6", "Blues")
    dataset.heatmap(D6[:, L==1], "raw_pos6", "Greens")
    dataset.heatmap(D6[:, L==0], "raw_neg6", "Reds")
    
    D5, _P = dataset.PCA_reduce(D, 5)
    dataset.heatmap(D5, "raw_m5", "Blues")
    dataset.heatmap(D5[:, L==1], "raw_pos5", "Greens")
    dataset.heatmap(D5[:, L==0], "raw_neg5", "Reds")
    
    print("done")
    print("Means of the 8 features:")
    print(mu)
    print("Variances of the 8 features:")
    print(sigma)

main()
