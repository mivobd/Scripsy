## Housekeeping
from types import SimpleNamespace
import json
import subprocess
import os
import numpy as np
from scipy.stats import f, norm, binom
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
import shutil
from pathlib import Path

subprocess.run('cls', shell=True)
timer_start = time.perf_counter()

# def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-4, file = ""):
#     """
#     Expectation-Maximization algorithm for Bayesian PCA

#     Parameters
#         x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
#         M : int, number of retained principal components
#         seed : int (optional), set to zero for reproducibility, otherwise set to any integer (preferably the counter in the global maximum search)
#         ads : int (optional), 0 for no automatic dimensionality selection (ADS) which returns normal the probabalistic PCA solution, 1 to enable ADS which returns the Bayesian PCA solution
#         limit : int (optional), the maximum allowable iterations
#         diff : float (optional), the maximum allowable difference between the current and previous iteration of the parameters W and beta

#     Returns
#         W_new : numpy array [DxM], the final estimate of the weight matrix
#         beta_new : float, the final estimate of the precision of the noise
#         hyper_new : numpy array [Mx1], the final estimate of the hyperparameters
#         count : int, the number of iterations performed
#     """
#     D, N = np.shape(x)
#     count = 0
#     hyper = 0
#     W = 0
#     beta = 0
#     E_z = np.zeros([M, N])
#     E_zzT = np.zeros([M, M, N])

#     np.random.seed(seed)
#     hyper_new = np.ones([M]).reshape(-1, 1)
#     W_new = np.random.rand(D,M)/np.random.rand(D,M)
#     beta_new = 1/(np.random.rand()/np.random.rand())

#     W_diff = np.sum((W_new-W)**2)
#     sigma2_diff = (beta_new-beta)**2
    
#     while count < limit and (W_diff > diff or sigma2_diff > diff ):
#         printProgressBar(count, limit, suffix = f"Complete ({file})")
#         # Updating parameters
#         W = W_new.copy()
#         beta = beta_new
#         W_sum1 = 0
#         W_sum2 = 0
#         I_M = np.identity(M)
#         M_matrix = W.T @ W + I_M/beta
#         M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)

#         # Expecation
#         for n in range(0, N):
#             E_z[:, n:n+1] = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT[:, :, n] = M_inv/beta + E_z[:, n:n+1] @ E_z[:, n:n+1].T

#             W_sum1 += x[:, n:n+1] @ E_z[:, n:n+1].T
#             W_sum2 += E_zzT[:, :, n]


#         # Maximisation
#         with np.errstate(divide='ignore', over='ignore'):
#             for i in range(0, M):
#                 hyper_new[i:i+1] = D /( W[:, i:i+1].T @ W[:, i:i+1] )
#                 # hyper_new[i:i+1] = D/(W[i:i+1, :]@W[i:i+1, :].T)

#         W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads/beta*np.diag(hyper_new.reshape(M)) )

#         beta_new = 0
#         for n in range(0, N):
#             beta_new += N*D / ( x[:, n:n+1].T @ x[:, n:n+1] - 2 * E_z[:, n:n+1].T @ W_new.T @ x[:, n:n+1] + np.trace( E_zzT[:, :, n] @ W_new.T @ W_new) )

#         W_diff = np.sum((W_new-W)**2)
#         beta_diff = (beta_new-beta)**2
#         count += 1

#     return [W_new, beta_new, hyper_new, count, seed]

# def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-4, file = ""):
#     """
#     next to try using
#     linalg.solve(B.T, A.T).T
#     to find A@inv(B) instead of using cholesky
#     """
#     count = 0
#     D, N = np.shape(x)
#     # np.random.seed(seed)

#     W = 0
#     W_new = np.random.rand(D,M)/10
#     # W_new = np.ones([D,M])
#     W_diff = np.sum((W_new-W)**2)

#     beta = 0
#     beta_new = (np.random.rand()/np.random.rand())*100
#     # beta_new = 1
#     beta_diff = (beta_new-beta)**2

#     A = 0
#     A_new = np.diag(np.ones([M]))
#     A_diff = np.sum((A_new-A)**2)

#     while count < limit and ( max(W_diff, beta_diff, A_diff) > diff ):
#         printProgressBar(count, limit, suffix = f"Complete ({file})")
#         W = W_new.copy()
#         beta = beta_new
#         A = A_new.copy()

#         I_M = np.identity(M)
#         M_matrix = W.T @ W + I_M/beta
#         # M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)
#         M_inv = np.linalg.solve(M_matrix.T, I_M.T).T

#         W_sum1 = 0
#         W_sum2 = 0
#         for n in range(0, N):
#             E_z = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT = M_inv/beta + E_z @ E_z.T

#             W_sum1 += x[:, n:n+1] @ E_z.T
#             W_sum2 += E_zzT
#         # W_new = W_sum1 @ la.cho_solve(la.cho_factor(W_sum2 + A/beta, lower=True), I_M)
#         W_new = np.linalg.solve((W_sum2 + A/beta).T, W_sum1.T).T
#         # print(W_new - W_new1)

#         beta_sum1 = 0
#         beta_sum2 = 0
#         beta_sum3 = 0
#         for n in range(0, N):
#             E_z = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT = M_inv/beta + E_z @ E_z.T

#             beta_sum1 += x[:, n] @ x[:, n]
#             beta_sum2 += E_z.T @ W_new.T @ x[:, n:n+1]
#             beta_sum3 += np.trace( E_zzT @ W_new.T @ W_new)
#         # print(beta_sum1)
#         beta_new = N*D / ( beta_sum1 - 2 * beta_sum2 + beta_sum3 )

#         for i in range(0, M):
#             A_new[i, i] = D / ( W_new[:, i] @ W_new[:, i] )

#         W_diff = np.sum((W_new-W)**2)
#         beta_diff = (beta_new-beta)**2
#         A_diff = np.sum((A_new-A)**2)
#         count += 1

#         # print(W_diff)
#         # print(beta_diff)
#         # print(A_diff)

#     return [W_new, beta_new, A_new, count, seed]

# def log_likelihood(x, parameters):
#     """
#     Computes the log-likelihood of the complete data given the parameters of the model.

#     Parameters
#         x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
#         parameters : list, containing the weight matrix W and the variance of the noise beta in the first and second elements, respectively

#     Returns
#         log_likelihood_value : float, the log-likelihood of the complete data given the parameters (single value but calculated as a numpy array, so sum is used so it is returned as a float)
#     """
#     W = parameters[0]
#     beta = parameters[1]
#     D, N = np.shape(x)
#     M = np.shape(W)[1]
#     x_mean = np.mean(x, axis=1).reshape(-1, 1)
#     I_M = np.identity(M)
#     M_matrix = W.T @ W + I_M/beta
#     M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)

#     E_z = np.zeros([M, N])
#     E_zzT = np.zeros([M, M, N])
#     log_sum1 = 0
#     log_sum2 = 0
#     log_sum3 = 0
#     log_sum4 = 0

#     for n in range(0, N):
#         E_z[:, n:n+1] = (M_inv@W.T)@x[:, n:n+1]
#         E_zzT[:, :, n] = M_inv/beta + E_z[:, n:n+1]@E_z[:, n:n+1].T

#         log_sum1 += np.trace(E_zzT[:, :, n])
#         log_sum2 += x[:, n:n+1].T@x[:, n:n+1]
#         log_sum3 +=  E_z[:, n:n+1].T @ W.T @ x[:, n:n+1]
#         log_sum4 += np.trace( (E_zzT[:, :, n] @ W.T) @ W)

#     log_liklihood_value = - 1/2 * ( N*D * np.log( 2*np.pi/beta ) + log_sum1 + beta * ( log_sum2 - 2 * log_sum3 + log_sum4) )

#     return np.sum(log_liklihood_value)


test = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
I_test = np.identity(3)
print(np.linalg.inv(test))
# print(la.cho_solve(la.cho_factor(test, lower=True), I_test))
print(np.linalg.solve(test, I_test))

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")