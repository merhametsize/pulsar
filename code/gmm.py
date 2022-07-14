#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 2 11:12:10 2022

@author: gabri
"""

import numpy
import scipy
import scipy.special

def column(v):
    return v.reshape((v.size), 1)

def row(v):
    return v.reshape(1, v.size)

class GMM_classifier:
    def __init__(self, n, mode, tiedness):
        self.n = n #number doublings in LBG, the number of components will be 2^n
        self.gmm0 = None
        self.gmm1 = None
        self.mode = mode
        self.tiedness = tiedness
    
    def train(self, DTR, LTR):
        D0 = DTR[:, LTR==0]
        D1 = DTR[:, LTR==1]
        
        self.gmm0 = self._LBG(D0, self.n)
        self.gmm1 = self._LBG(D1, self.n)
    
    def compute_lls(self, DTE):
        ll0 = self._GMM_ll_per_sample(DTE, self.gmm0)
        ll1 = self._GMM_ll_per_sample(DTE, self.gmm1)
        return ll0, ll1
    
    def compute_scores(self, DTE):
        ll0, ll1 = self.compute_lls(DTE)
        return ll1-ll0
    
    def _LBG(self, X, doublings):
        initial_mu = column(X.mean(1))
        initial_sigma = numpy.cov(X)
        
        gmm = [(1.0, initial_mu, initial_sigma)]
        for i in range(doublings):
            doubled_gmm = []
            for component in gmm:
                w = component[0]
                mu = component[1]
                sigma = component[2]
                
                U, s, Vh = numpy.linalg.svd(sigma)
                d = U[:, 0:1] * s[0]**0.5 * 0.1
                component1 = (w/2, mu+d, sigma)
                component2 = (w/2, mu-d, sigma)
                doubled_gmm.append(component1)
                doubled_gmm.append(component2)
            if self.mode == "full" and self.tiedness == "untied":
                gmm = self._GMM_EM(X, doubled_gmm)
            elif self.mode == "naive" and self.tiedness == "untied":
                gmm = self._GMM_EM_diag(X, doubled_gmm)
            elif self.mode == "full" and self.tiedness == "tied":
                gmm = self._GMM_EM_tied(X, doubled_gmm)
            elif self.mode == "naive" and self.tiedness == "tied":
                gmm = self._GMM_EM_diag_tied(X, doubled_gmm)
        return gmm
    
    def _logpdf_GAU_ND_Opt(self, X, mu, C):
        P = numpy.linalg.inv(C)
        const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
        const += -0.5 * numpy.linalg.slogdet(C)[1]
        
        Y = []
        for i in range(X.shape[1]):
            x = X[:, i:i+1]
            res = const + -0.5 * numpy.dot((x-mu).T, numpy.dot(P, (x-mu)))
            Y.append(res)
        
        return numpy.array(Y).ravel()

    def _GMM_ll_per_sample(self, X, gmm):
        G = len(gmm)
        N = X.shape[1]
        S = numpy.zeros((G, N))
        
        for g in range(G):
            S[g, :] = self._logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        return scipy.special.logsumexp(S, axis=0)

    def _GMM_EM(self, X, gmm):
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self._logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (row(gamma)*X).sum(1)
                S = numpy.dot(X, (row(gamma)*X).T)
                w = Z/N
                mu = column(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                #constraint
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, column(s)*U.T)
                
                gmm_new.append((w, mu, sigma))
            gmm = gmm_new
            #print(ll_new)
        #print(ll_new-ll_old)
        return gmm
    
    def _GMM_EM_diag(self, X, gmm):
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self._logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (row(gamma)*X).sum(1)
                S = numpy.dot(X, (row(gamma)*X).T)
                w = Z/N
                mu = column(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                #diagonalization
                sigma = sigma * numpy.eye(sigma.shape[0])
                
                #constraint
                U, s, _ = numpy.linalg.svd(sigma)
                s[s<psi] = psi
                sigma = numpy.dot(U, column(s)*U.T)
                
                gmm_new.append((w, mu, sigma))
            gmm = gmm_new
            #print(ll_new)
        #print(ll_new-ll_old)
        return gmm
    
    def _GMM_EM_tied(self, X, gmm):
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self._logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            summatory = numpy.zeros((X.shape[0], X.shape[0]))
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (row(gamma)*X).sum(1)
                S = numpy.dot(X, (row(gamma)*X).T)
                w = Z/N
                mu = column(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                summatory += Z*sigma
                gmm_new.append((w, mu, sigma))
            #tying
            sigma = summatory / G
            #constraint
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, column(s)*U.T)
            gmm = gmm_new
            #print(ll_new)
        #print(ll_new-ll_old)
        return gmm
    
    def _GMM_EM_diag_tied(self, X, gmm):
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = self._logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            summatory = numpy.zeros((X.shape[0], X.shape[0]))
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (row(gamma)*X).sum(1)
                S = numpy.dot(X, (row(gamma)*X).T)
                w = Z/N
                mu = column(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                #diagonalization
                sigma = sigma * numpy.eye(sigma.shape[0])
                summatory += Z*sigma
                gmm_new.append((w, mu, sigma))
            #tying
            sigma = summatory / G
            #constraint
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, column(s)*U.T)
            gmm = gmm_new
            #print(ll_new)
        #print(ll_new-ll_old)
        return gmm
