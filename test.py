from types import SimpleNamespace
import json
import os
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

os.system('cls')

def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in k loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def hinton(matrix, ax=None):
    """
    Draws the Hinton diagram for visualizing a weight matrix.

    Parameters
        matrix : numpy array, the weight matrix to be visualized
        ax : matplotlib axes object (optional), use if you want to plot on a specific set of axes

    Returns
        None
    """ 
    ax = ax if ax is not None else plt.gca()
    matrix = matrix.T # otherwise plots rows as columns and columns as rows
    height, width = np.shape(matrix)

    max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        if w == 0:
            w = 1e-8
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    # ax.set_xlim(-0.5, width - 0.5)
    # ax.set_ylim(-0.5, height - 0.5)
    ax.autoscale_view()
    ax.invert_yaxis()
    # plt.show()

# def em(x, ads = 1, limit = 1000, diff = 1e-3):
#     D, N = np.shape(x)
#     x_mean = np.mean(x, axis=1).reshape(-1, 1)
#     count = 0
#     hyper = 0
#     W = 0
#     sigma2 = 0
#     E_z = np.zeros([D, N])
#     E_zzT = np.zeros([D, D, N])

#     U, S, Vt = np.linalg.svd(x, full_matrices=False)
#     # np.random.seed(int(time.time()))
#     np.random.seed(1)
#     hyper_new = np.ones([D]).reshape(-1, 1)
#     W_new = U#np.random.rand(D,D)/np.random.rand(D,D)
#     sigma2_new = np.random.rand()/np.random.rand()

#     W_diff = np.amax(np.abs((W_new-W)))**2
#     sigma2_diff = (sigma2_new-sigma2)**2
#     hyper_diff = np.amax((hyper_new-hyper)**2)
    
#     while count < limit and ( W_diff > diff or sigma2_diff > diff ):
#         # Updating parameters
#         W = W_new
#         sigma2 = sigma2_new
#         W_sum1 = 0
#         W_sum2 = 0
#         sigma2_sum = 0
#         E_zzT_sum = 0
#         M_inv = np.linalg.inv( W.T @ W + sigma2 * np.identity(D) )

#         # Expecation
#         for n in range(0, N):
#             E_z[:, n:n+1] = (M_inv@W.T)@(x[:, n:n+1]-x_mean)
#             E_zzT[:, :, n] = sigma2*M_inv.T + E_z[:, n:n+1]@E_z[:, n:n+1].T

#             W_sum1 += (x[:, n:n+1]-x_mean)@E_z[:, n:n+1].T
#             W_sum2 += E_zzT[:, :, n]

#         # E_zzT_sum = np.sum(E_zzT, axis=2)

#         # Maximisation
#         for i in range(0, D):
#             # hyper_new[i:i+1] = D/(W[:, i:i+1].T@W[:, i:i+1])
#             hyper_new[i:i+1] = D/(W[i:i+1, :]@W[i:i+1, :].T)

#         # W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads*sigma2*np.diag(hyper_new.reshape(D)) )
#         W_new = np.linalg.inv( W_sum2 + ads*sigma2*np.diag(hyper_new.reshape(D)) ) @ W_sum1

#         # W_new = ((x-x_mean) @ E_z.T ) @ np.linalg.inv( E_zzT_sum + ads*sigma2*np.diag(hyper_new.reshape(D)) )
#         # W_new = (E_z @ (x-x_mean).T ) @ np.linalg.inv( E_zzT_sum + ads*sigma2*np.diag(hyper_new.reshape(D)) )

#         sigma2_new = 0
#         for n in range(0, N):
#             sigma2_new += 1/(N*D) * ( (x[:, n:n+1]-x_mean).T@(x[:, n:n+1]-x_mean) - 2 * E_z[:, n:n+1].T @ W_new @ (x[:, n:n+1]-x_mean) + np.trace( (E_zzT[:, :, n] @ W_new) @ W_new.T) )

#         W_diff = np.sum(np.abs((W_new-W))**2)
#         sigma2_diff = (sigma2_new-sigma2)**2
#         count += 1
#         printProgressBar(count, limit)

#     return W_new, sigma2_new, hyper_new, count

# def em_row(x, M, ads = 1, limit = 1000, diff = 1e-3):
#     D, N = np.shape(x)
#     x_mean = np.mean(x, axis=1).reshape(-1, 1)
#     count = 0
#     hyper = 0
#     W = 0
#     sigma2 = 0
#     E_z = np.zeros([N, M])
#     E_zzT = np.zeros([N, M, M])

#     # U, S, Vt = np.linalg.svd(x, full_matrices=False)
#     # np.random.seed(int(time.time()))
#     np.random.seed(1)
#     hyper_new = np.ones([M])
#     W_new = np.random.rand(D,M)/np.random.rand(D,M)
#     sigma2_new = np.random.rand()/np.random.rand()

#     W_diff = np.amax(np.abs((W_new-W)))**2
#     sigma2_diff = (sigma2_new-sigma2)**2
#     hyper_diff = np.amax((hyper_new-hyper)**2)
    
#     while count < limit and ( W_diff > diff or sigma2_diff > diff ):
#         # Updating parameters
#         W = W_new
#         sigma2 = sigma2_new
#         W_sum1 = 0
#         W_sum2 = 0
#         sigma2_sum = 0
#         E_zzT_sum = 0
#         M_inv = np.linalg.inv( W.T @ W + sigma2 * np.identity(M) )

#         # Expecation
#         for n in range(0, N):
#             # E_z[n:n+1, :] = (M_inv@W.T)@(x[:, n:n+1]-x_mean)
#             E_z[n:n+1, :] = (W.T@(x[:, n:n+1]-x_mean)).T@M_inv
#             E_zzT[n, :, :] = sigma2*M_inv.T + E_z[n:n+1, :]@E_z[n:n+1, :].T

#             W_sum1 += (x[:, n:n+1]-x_mean)@E_z[n:n+1, :]
#             W_sum2 += E_zzT[n, :, :]

#         # Maximisation
#         for i in range(0, M):
#             # hyper_new[i:i+1] = D/(W[:, i:i+1].T@W[:, i:i+1])
#             hyper_new[i:i+1] = D/(W[i:i+1, :]@W[i:i+1, :].T)

#         # W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads*sigma2*np.diag(hyper_new.reshape(D)) )
#         W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads*sigma2*np.diag(hyper_new.reshape(M)) )

#         # W_new = ((x-x_mean) @ E_z.T ) @ np.linalg.inv( E_zzT_sum + ads*sigma2*np.diag(hyper_new.reshape(D)) )
#         # W_new = (E_z @ (x-x_mean).T ) @ np.linalg.inv( E_zzT_sum + ads*sigma2*np.diag(hyper_new.reshape(D)) )

#         sigma2_new = 0
#         for n in range(0, N):
#             sigma2_new += 1/(N*D) * ( (x[:, n:n+1]-x_mean).T@(x[:, n:n+1]-x_mean) - 2 * E_z[n:n+1, :] @ W_new.T @ (x[:, n:n+1]-x_mean) + np.trace( (E_zzT[n, :, :] @ W_new.T) @ W_new) )

#         W_diff = np.sum(np.abs((W_new-W))**2)
#         sigma2_diff = (sigma2_new-sigma2)**2
#         count += 1
#         printProgressBar(count, limit)

#     return W_new, sigma2_new, hyper_new, count

def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-6):
    """
    Expectation-Maximization algorithm for Bayesian PCA

    Parameters
        x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
        M : int, number of retained principal components
        seed : int (optional), set to zero for reproducibility, otherwise set to any integer (preferably the counter in the global maximum search)
        ads : int (optional), 0 for no automatic dimensionality selection (ADS) which returns normal the probabalistic PCA solution, 1 to enable ADS which returns the Bayesian PCA solution
        limit : int (optional), the maximum allowable iterations
        diff : float (optional), the maximum allowable difference between the current and previous iteration of the parameters W and sigma2

    Returns
        W_new : numpy array [DxM], the final estimate of the weight matrix
        sigma2_new : float, the final estimate of the variance of the noise
        hyper_new : numpy array [Mx1], the final estimate of the hyperparameters
        count : int, the number of iterations performed
    """
    D, N = np.shape(x)
    count = 0
    hyper = 0
    W = 0
    sigma2 = 0
    E_z = np.zeros([M, N])
    E_zzT = np.zeros([M, M, N])

    np.random.seed(seed)
    hyper_new = np.ones([M]).reshape(-1, 1)
    W_new = np.random.rand(D,M)/np.random.rand(D,M)
    sigma2_new = np.random.rand()/np.random.rand()

    W_diff = np.sum((W_new-W)**2)
    sigma2_diff = (sigma2_new-sigma2)**2
    
    while count < limit and (W_diff > diff or sigma2_diff > diff):
        # Updating parameters
        W = W_new.copy()
        sigma2 = sigma2_new
        W_sum1 = 0
        W_sum2 = 0
        M_inv = np.linalg.inv( W.T @ W + sigma2 * np.identity(M) )

        # Expecation
        for n in range(0, N):
            E_z[:, n:n+1] = M_inv @ W.T @ x[:, n:n+1]
            E_zzT[:, :, n] = sigma2 * M_inv + E_z[:, n:n+1] @ E_z[:, n:n+1].T

            W_sum1 += x[:, n:n+1] @ E_z[:, n:n+1].T
            W_sum2 += E_zzT[:, :, n]


        # Maximisation
        with np.errstate(divide='ignore', over='ignore'):
            for i in range(0, M):
                hyper_new[i:i+1] = D /( W[:, i:i+1].T @ W[:, i:i+1] )
                # hyper_new[i:i+1] = D/(W[i:i+1, :]@W[i:i+1, :].T)

        W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads*sigma2*np.diag(hyper_new.reshape(M)) )

        sigma2_new = 0
        for n in range(0, N):
            sigma2_new += 1/(N*D) * ( x[:, n:n+1].T @ x[:, n:n+1] - 2 * E_z[:, n:n+1].T @ W_new.T @ x[:, n:n+1] + np.trace( E_zzT[:, :, n] @ W_new.T @ W_new) )

        W_diff = np.sum((W_new-W)**2)
        sigma2_diff = (sigma2_new-sigma2)**2
        count += 1
        # printProgressBar(count, limit)

    return W_new, sigma2_new, hyper_new, count, seed

def log_likelihood(x, parameters):
    """
    Computes the log-likelihood of the complete data given the parameters of the model.

    Parameters
        x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
        parameters : list, containing the weight matrix W and the variance of the noise sigma2 in the first and second elements, respectively

    Returns
        log_likelihood_value : float, the log-likelihood of the complete data given the parameters (single value but calculated as a numpy array, so sum is used so it is returned as a float)
    """
    W = parameters[0]
    sigma2 = parameters[1]
    D, N = np.shape(x)
    M = np.shape(W)[1]
    x_mean = np.mean(x, axis=1).reshape(-1, 1)
    M_inv = np.linalg.inv( W.T @ W + sigma2 * np.identity(M) )

    E_z = np.zeros([M, N])
    E_zzT = np.zeros([M, M, N])
    log_sum1 = 0
    log_sum2 = 0
    log_sum3 = 0
    log_sum4 = 0

    for n in range(0, N):
        E_z[:, n:n+1] = (M_inv@W.T)@x[:, n:n+1]
        E_zzT[:, :, n] = sigma2*M_inv.T + E_z[:, n:n+1]@E_z[:, n:n+1].T

        log_sum1 += np.trace(E_zzT[:, :, n])
        log_sum2 += x[:, n:n+1].T@x[:, n:n+1]
        log_sum3 +=  E_z[:, n:n+1].T @ W.T @ x[:, n:n+1]
        log_sum4 += np.trace( (E_zzT[:, :, n] @ W.T) @ W)

    log_liklihood_value = - 1/2 * ( N*D * np.log( 2*np.pi*sigma2 ) + log_sum1 + 1/(sigma2) * ( log_sum2 - 2 * log_sum3 + log_sum4) )

    return np.sum(log_liklihood_value)

# x = np.array([[7, 26, 6, 60], [1, 29, 15, 52], [11, 56, 8, 20], [11, 31, 8, 47], [7, 52, 6, 33], [11, 55, 9, 22], [3, 71, 17, 6], [1, 31, 22, 44], [2, 54, 18, 22], [21, 47, 4, 26], [1, 40, 23, 34], [11, 66, 9, 12], [10, 68, 8, 12]], dtype=np.float64)
# x = x.T

x = np.zeros([10, 300])
np.random.seed(1)

for d in range(0, 10):
    x[d, :] = np.random.normal(loc = 0, scale = 0.5, size = 300)
for d in [1, 3, 6]:
    x[d, :] = np.random.normal(loc = 0, scale = 1, size = 300)

x = x - np.mean(x, axis=1).reshape(-1, 1) # centering the data

limit = 1
best_log_likelihood = -1e10
for iteration in range(0, limit):
    seconds = time.time()*1e4
    seed = int((seconds - int(seconds))*10000)
    parameters = em(x, 9, iteration+1)
    log_likelihood_value = log_likelihood(x, parameters)

    if log_likelihood_value > best_log_likelihood:
        best_log_likelihood = log_likelihood_value
        best_parameters = parameters

    if parameters[3] == 1e4:
        print("EM algorithm did not converge within the limit of iterations")
        break

    printProgressBar(iteration, limit)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ppca = em(x, 9, best_parameters[4], False)
hinton(ppca[0], ax1)
ax1.set_title("PPCA")
ax1.set_xlabel("Features")
ax1.set_ylabel("Directions")
print("\nPPCA ", log_likelihood(x, ppca))
hinton(best_parameters[0], ax2)
ax2.set_title("BPCA")
ax2.set_xlabel("Features")
print("\nBPCA ", best_log_likelihood)
plt.show()

# seconds = time.time()*1e4
# seed = int((seconds - int(seconds))*10000)
# W_ppca = em(x, 9, seed, False)
# W_bpca = em(x, 9, seed, True)
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
# hinton(W_ppca[0], ax=ax1)
# hinton(W_bpca[0], ax=ax2)
# plt.show()
