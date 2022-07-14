#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:01:13 2022

@author: gabri
"""

import numpy
from scipy.stats import norm, rankdata
import matplotlib.pyplot as plt

def column(v):
    return v.reshape((v.size), 1)

def load_data(file):
    matrix = []
    labels = []
    lines = open(file, "r")
    for l in lines:
        tokens = l.split(",")
        attrVector = column(numpy.array(tokens[0:-1], dtype=float))
        label = tokens[-1]

        labels.append(label)
        matrix.append(attrVector)
    D = numpy.hstack(matrix)
    L = numpy.array(labels, dtype=numpy.int32)
    return D, L
    
def plot_histograms(D, L, title):
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    features = \
    {
        0: 'Mean of the integrated profile',
        1: 'Standard deviation of the integrated profile',
        2: 'Excess kurtosis of the integrated profile',
        3: 'skewness of the integrated profile',
        4: 'Mean of the DM-SNR curve',
        5: 'Standard deviation of the DM-SNR curve',
        6: 'Excess kurtosis of the DM-SNR curve',
        7: 'skewness of the DM-SNR curve'
    }

    for f in range(len(features)):
        plt.figure()
        plt.xlabel(features[f])
        plt.hist(D0[f, :], bins = 25, density = True, alpha = 0.4, label = 'Interference')
        plt.hist(D1[f, :], bins = 25, density = True, alpha = 0.4, label = 'Pulsar')

        plt.legend()
        plt.tight_layout()
        plt.savefig('hist_%s%d.pdf' % (title, f))

def gaussianize(DTR, DTE=None):
    rankDTR = numpy.zeros(DTR.shape)
    for i in range(DTR.shape[0]):
        for j in range(DTR.shape[1]):
            rankDTR[i][j] = (DTR[i] < DTR[i][j]).sum()
    rankDTR = (rankDTR+1) / (DTR.shape[1]+2)
    
    if(DTE is not None):
        rankDTE = numpy.zeros(DTE.shape)
        for i in range(DTE.shape[0]):
            for j in range(DTE.shape[1]):
                rankDTE[i][j] = (DTR[i] < DTE[i][j]).sum()
        rankDTE = (rankDTE+1) / (DTR.shape[1]+2)
        return norm.ppf(rankDTR), norm.ppf(rankDTE)
    return norm.ppf(rankDTR)

def gaussianify_DTR(X):
    return norm.ppf(rankdata(X, method='average', axis=1) / (X.shape[1] + 2))
    
def normalize_zscore(D, mu=[], sigma=[]):
    if mu == [] or sigma == []:
        mu = numpy.mean(D, axis=1)
        sigma = numpy.std(D, axis=1)
    ZD = D
    ZD = ZD - column(mu)
    ZD = ZD / column(sigma)
    return ZD, mu, sigma

def PCA_reduce(D, m):
    mu = D.mean(1)
    DC = D - column(mu)
    C = numpy.dot(DC, DC.T) / float(DC.shape[1])
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    #U, s, Vh = numpy.linalg.svd(C)
    #P = U[:, 0:m]

    # desc_eigenvalues = s[::-1]
    # eig_total = s.sum()
    # sum = 0
    # for i in range(0, 4):
    #     sum += desc_eigenvalues[i]
    #     print(sum / eig_total * 100)

    PCA_D = numpy.dot(P.T, D)
    return PCA_D, P

def heatmap(D, title, color):
    plt.figure()
    pearson_matrix = numpy.corrcoef(D)
    plt.imshow(pearson_matrix, cmap=color, vmin=-1, vmax=1)
    plt.savefig("heatmap_%s.pdf" % (title))

def plot_scatter(D, L, title):
    D, _ = PCA_reduce(D, 2)
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    plt.figure()
    plt.title(title)
    plt.scatter(D0[0, :], D0[1, :], label = 'Interference')
    plt.scatter(D1[0, :], D1[1, :], label = 'Pulsars')
    plt.legend()
    plt.tight_layout()
    plt.savefig('PCA_%s.pdf' % title)

def plot_show():
    plt.show()

