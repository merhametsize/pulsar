#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:44:27 2022

@author: gabri
"""

import numpy
import scipy

def column(v):
    return v.reshape((v.size), 1)

def row(v):
    return v.reshape(1, v.size)

class SupportVectorMachines:
    def __init__(self, C, mode, pT, gamma=1, d=2, K=1):
        self.C = C
        self.mode = mode
        self.pT = pT
        self.d = d
        self.gamma = gamma
        self.K = K
        self.w_start = None
        self.H = None
    
    def train(self, DTR, LTR):
        DTRext = numpy.vstack([DTR, numpy.ones((1, DTR.shape[1]))])
        
        DTR0 = DTR[:, LTR==0]
        DTR1 = DTR[:, LTR==1]
        nF = DTR0.shape[1]
        nT = DTR1.shape[1]
        emp_prior_F = (nF / DTR.shape[1])
        emp_prior_T =  (nT / DTR.shape[1])
        Cf = self.C * self.pT / emp_prior_F
        Ct = self.C * self.pT / emp_prior_T
    
        Z = numpy.zeros(LTR.shape)
        Z[LTR == 0] = -1
        Z[LTR == 1] = 1
        
        if self.mode == "linear":
            H = numpy.dot(DTRext.T, DTRext)
            H = column(Z) * row(Z) * H
        elif self.mode == "polynomial":
            H = numpy.dot(DTRext.T, DTRext) ** self.d
            H = column(Z) * row(Z) * H
        elif self.mode == "RBF":
            dist = column((DTR**2).sum(0)) + row((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
            H = numpy.exp(-self.gamma * dist) + self.K
            H = column(Z) * row(Z) * H
        
        self.H = H
        
        bounds = [(-1, -1)] * DTR.shape[1]
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bounds[i] = (0, Cf)
            else:
                bounds[i] = (0, Ct)
        
        alpha_star, x, y = scipy.optimize.fmin_l_bfgs_b(
            self._LDual, 
            numpy.zeros(DTR.shape[1]),
            #bounds = [(0, self.C)] * DTR.shape[1],
            bounds = bounds,
            factr = 1e7,
            maxiter = 100000,
            maxfun = 100000
                )

        self.w_star = numpy.dot(DTRext, column(alpha_star) * column(Z))
    
    def compute_scores(self, DTE):
        DTEext = numpy.vstack([DTE, numpy.ones((1, DTE.shape[1]))])
        S = numpy.dot(self.w_star.T, DTEext)
        return S
        
    def _JDual(self, alpha):
        Ha = numpy.dot(self.H, column(alpha))
        aHa = numpy.dot(row(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def _LDual(self, alpha):
        loss, grad = self._JDual(alpha)
        return -loss, -grad
    
    def _JPrimal(self, DTRext, w, Z):
        S = numpy.dot(row(w), DTRext)
        loss = numpy.maximum(numpy.zeros(S.shape), 1-Z*S).sum()
        return 0.5*numpy.linalg.norm(w)**2 + self.C*loss
