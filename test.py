## Housekeeping
from types import SimpleNamespace
import json
import subprocess
import os
import numpy as np
from scipy.stats import f, norm, binom, chi2
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
import shutil
from pathlib import Path
if not hasattr(np, "float128"):
    np.float128 = np.longdouble
from pyppca import ppca

subprocess.run('cls', shell=True)
timer_start = time.perf_counter()

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
    if type(matrix) == list:
        matrix = np.array(matrix)
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



def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-4, file = ""):
    """
    next to try using
    linalg.solve(B.T, A.T).T
    to find A@inv(B) instead of using cholesky
    """
    count = 0
    D, N = np.shape(x)
    I_M = np.identity(M)
    np.random.seed(seed)

    W = 0
    W_new = np.random.rand(D,M)/10
    # W_new = np.ones([D,M])
    W_diff = np.sum((W_new-W)**2)

    beta = 0
    beta_new = (np.random.rand()/np.random.rand())*100
    # beta_new = 1
    beta_diff = (beta_new-beta)**2

    alphas = 0
    alphas_new = np.ones([M])
    alphas_diff = np.sum((alphas_new-alphas)**2)

    M_matrix = 0
    M_inv = 0
    E_z = 0
    sum_E_zzT = 0

    total_time = 0

    while count < limit and ( max(W_diff, alphas_diff, beta_diff) > diff ):
        last_start = time.perf_counter()
        printProgressBar(count, limit, prefix = f'Iteration {count} | W_diff = {W_diff} | beta_diff = {beta_diff} | alphas_diff = {alphas_diff}', suffix = f"Maximum allowed       ", length = 50)
        W = W_new.copy()
        beta = beta_new
        alphas = alphas_new.copy()

        # M_matrix += W.T @ W + I_M/beta
        M_matrix = W.T@W + I_M/beta
        # M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)
        M_inv = np.linalg.solve(M_matrix, I_M)

        # Expectation
        E_z = M_inv @ W.T @ x
        sum_E_zzT = N * M_inv / beta + E_z @ E_z.T

        # Maximisation
        W_new = np.linalg.solve( sum_E_zzT + ads * np.diag(alphas/beta).T, (x @ E_z.T).T).T

        beta_new = N*D / ( np.trace(x.T @ x) - 2 * np.trace(E_z.T @ W_new.T @ x) + np.trace(W_new.T @ W_new @ sum_E_zzT) )

        alphas_new = D / np.sum(W_new*W_new, axis=0)

        # print("alpha:", alphas_new)
        # print("1 / alpha:", 1 / alphas_new)
        # print("||w_i||^2:", np.sum(W_new*W_new, axis=0))

        # Convergence checks
        # temp = W_new-W
        # W_diff = np.where(-temp.min() > temp.max(), temp.min(), temp.max())
        # W_diff *= W_diff
        # # print(W_diff)
        # beta_diff = np.abs(beta_new-beta)/min(beta_new, beta)
        # # print(beta_diff)
        # temp = alphas_new-alphas
        # alphas_diff = np.where(-temp.min() > temp.max(), temp.min(), temp.max())
        # alphas_diff *= alphas_diff
        # # print(A_diff)

        W_diff = (
            np.linalg.norm(W_new - W)
            / max(np.linalg.norm(W), 1.0)
        )

        beta_diff = (
            abs(beta_new - beta)
            / max(abs(beta_new), abs(beta), 1.0)
        )

        mask = np.isfinite(alphas_new) & np.isfinite(alphas)
        alphas_diff = (
            np.linalg.norm(alphas_new[mask] - alphas[mask])
            / max(np.linalg.norm(alphas[mask]), 1.0)
        )

        # print(
        #     f"count={count}, "
        #     f"W_diff={W_diff}, "
        #     f"beta_diff={beta_diff}, "
        #     f"alphas_diff={alphas_diff}"
        # )

        count += 1

        recent_time = time.perf_counter() - last_start
        total_time += recent_time
        print(f"\rIteration {count} time: {recent_time} s (average {total_time/count} s)", end = "\r")

    # subprocess.run('cls', shell=True)
    return [W_new, beta_new, alphas_new, count, seed]



def log_likelihood(x, parameters):

    W = parameters[0]
    beta = parameters[1]
    A = parameters[2]
    D, N = np.shape(x)

    I_M = np.identity(M)
    M_matrix = W.T @ W + I_M/beta
    M_inv = np.linalg.solve( M_matrix, I_M)

    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    for n in range(0, N):
        E_z = M_inv @ W.T @ x[:, n:n+1]
        E_zzT = M_inv/beta + E_z @ E_z.T

        sum1 += np.trace(E_zzT)
        sum2 += x[:, n] @ x[:, n]
        sum3 += E_z.T @ W.T @ x[:, n]
        sum4 += np.trace(E_zzT @ W.T @ W)

    logliklihood = - np.sum((N*D*np.log(2*np.pi/beta) + sum1 + beta*sum2 - 2*beta*sum3 + beta*sum4)/2) # the sum is only here to ensure the return value is a float, not a numpy array. The expression inside the np.sum() returns 2D numpy array with one entry

    return logliklihood

def global_search(x, M, limit, save_state = False, file = ""):
    """
    Tries to find global maximum of the log-likelihood function.

    Parameters

    """
    state = SimpleNamespace()

    if save_state == False:
        state.best_log_likelihood = -1e10
        state.seed = np.zeros([limit], int).tolist()
        best_parameters = None
        state.log_likelihood_value = np.zeros([limit])
        state.log_likelihood_value = state.log_likelihood_value.tolist()
        state.iteration = 0
        state.bestIteration = 0

        for iteration in range(0, limit):
            seconds = time.time()*1e4
            state.seed[iteration] = int((iteration+1)*(seconds - int(seconds))*10000)
    elif os.path.exists("save_state.json") == False:
        raise Exception("save_state.json file not found. Please set save_state to False to start a new search.")
    else:
        with open("save_state.json", "r") as json_file:
            state = SimpleNamespace(**json.load(json_file))
        best_parameters = state.best_parameters

    for iteration in range(0 + state.iteration, limit):
        # printProgressBar(iteration, limit, suffix = f"Complete ({file})")
        completion = f"{file} [{(iteration+1)/limit*100} % overall]"
        parameters = em(x, M, state.seed[iteration], file = completion)
        state.log_likelihood_value[iteration] = log_likelihood(x, parameters)

        if state.log_likelihood_value[iteration] > state.best_log_likelihood:
            state.best_log_likelihood = state.log_likelihood_value[iteration]
            best_parameters = parameters
            state.bestIteration = iteration

        if parameters[3] == 1e4:
            print("EM algorithm did not converge within the limit of iterations")

        state.iteration = iteration
        state.best_parameters = best_parameters
        for index, item in enumerate(state.best_parameters):
            if type(item) == np.ndarray:
                state.best_parameters[index] = item.tolist()
                state.best_parameters[index]
        with open("save_state.json", "w") as json_file:
            json.dump(state.__dict__, json_file)

        # if state.bestIteration - iteration > 50:
        #     print("Global search has not found a better solution in 50 iterations, stopping search")
        #     break

    return state.best_log_likelihood, best_parameters



D = 10
N = 300
true_M = 3
lim = 0.05

num = SimpleNamespace()
stats = SimpleNamespace()
k = true_M
num.variables = D
num.measurements = N

with open("test_x.json", "r") as json_file:
    x = np.array(json.load(json_file))

W, sigma2, M, temp, Ye = ppca(x.T, true_M, 0)
print(np.shape(W))
beta = 1/sigma2
print(beta)

# results = SimpleNamespace()
# with open(f"test_results.json", 'r', encoding='utf-8') as file:
#     results.Snapshot = SimpleNamespace(**json.load(file))

# W_unordered = np.array(results.Snapshot.best_parameters[0])
# alphas = np.array(results.Snapshot.best_parameters[2])
# ordered_indices = np.argsort(1/alphas.reshape(-1))
# W = W_unordered[:, ordered_indices[::-1]]
# alphas_ordered = alphas[ordered_indices[::-1]]
# beta = results.Snapshot.best_parameters[1]

I_K = np.identity(k)
I_D = np.identity(num.variables)

# x = x.T
P = W[:, :k] # principal subspace
M = P.T @ P + I_K / beta
S_var_tx_inv = beta*M
t = P.T @ np.linalg.solve( P @ P.T + I_D / beta, x ) # score matrix for BPCA
stats.tsquared95 = chi2.ppf(1-lim, k) # Bayesian PCA version of 95 % confidence limit for Hotelling's T-squared 

# print("W_unordered:", W_unordered.shape)
# print("alphas:", alphas.shape)
# print("ordered_indices:", ordered_indices.shape)
# print("W:", W.shape)
# print("P:", P.shape)
# print("rank(P):", np.linalg.matrix_rank(P))
# print("cond(P.T @ P + I_K / beta):",
#       np.linalg.cond(P.T @ P + I_K / beta))

stats.tsquared = np.zeros([num.measurements])
for column in range(0, num.measurements):
    stats.tsquared[column] = (t[:, column].reshape(-1, 1).T @ S_var_tx_inv @ t[:, column])[0]

fig2, ax2 = plt.subplots(1, 1) # for T-squared contributions and T-squared statistic
ax2.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
ax2.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared, color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
# ax2.plot([num.ignore, num.ignore], [0, 2*stats.tsquared95], color = 'r', linestyle = '-', label = "Fault occurrence")
ax2.set_xlabel("Observation")
ax2.set_ylabel("Hotelling's T-squared statistic")
ax2.legend(loc = 'upper left')
fig2.show()
# print(f"Violations: {stats.tsquared_violations[kk]} out of {stats.r95}")
input("Press Enter to continue...")
ax2.cla()

results = SimpleNamespace()
with open(f"test_results.json", 'r', encoding='utf-8') as file:
    results.Snapshot = SimpleNamespace(**json.load(file))

W_unordered = np.array(results.Snapshot.best_parameters[0])
alphas = np.array(results.Snapshot.best_parameters[2])
ordered_indices = np.argsort(1/alphas.reshape(-1))
print(np.shape(W))
W = W_unordered[:, ordered_indices[::-1]]
alphas_ordered = alphas[ordered_indices[::-1]]
beta = results.Snapshot.best_parameters[1]
print(beta)

I_K = np.identity(k)
I_D = np.identity(num.variables)

# x = x.T
P = W[:, :k] # principal subspace
M = P.T @ P + I_K / beta
S_var_tx_inv = beta*M
t = P.T @ np.linalg.solve( P @ P.T + I_D / beta, x ) # score matrix for BPCA
stats.tsquared95 = chi2.ppf(1-lim, k) # Bayesian PCA version of 95 % confidence limit for Hotelling's T-squared 

stats.tsquared = np.zeros([num.measurements])
for column in range(0, num.measurements):
    stats.tsquared[column] = (t[:, column].reshape(-1, 1).T @ S_var_tx_inv @ t[:, column])[0]

fig2, ax2 = plt.subplots(1, 1) # for T-squared contributions and T-squared statistic
ax2.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
ax2.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared, color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
# ax2.plot([num.ignore, num.ignore], [0, 2*stats.tsquared95], color = 'r', linestyle = '-', label = "Fault occurrence")
ax2.set_xlabel("Observation")
ax2.set_ylabel("Hotelling's T-squared statistic")
ax2.legend(loc = 'upper left')
fig2.show()
# print(f"Violations: {stats.tsquared_violations[kk]} out of {stats.r95}")
input("Press Enter to continue...")
ax2.cla()

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")