## Housekeeping
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
timer_start = time.perf_counter()
# plt.ion()

# Print iterations progress
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
        printProgressBar(count, limit)

    return W_new, sigma2_new, hyper_new, count

def log_likelihood(x, parameters):
    """
    Computes the log-likelihood of the complete data given the parameters of the model.
    Parameters
        x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
        parameters : list, containing the weight matrix W and the variance of the noise sigma2 in the first and second elements, respectively
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
        E_z[:, n:n+1] = (M_inv@W.T)@(x[:, n:n+1])
        E_zzT[:, :, n] = sigma2*M_inv.T + E_z[:, n:n+1]@E_z[:, n:n+1].T

        log_sum1 += np.trace(E_zzT[:, :, n])
        log_sum2 += (x[:, n:n+1]).T@(x[:, n:n+1])
        log_sum3 +=  E_z[:, n:n+1].T @ W.T @ (x[:, n:n+1])
        log_sum4 += np.trace( (E_zzT[:, :, n] @ W.T) @ W)

    log_liklihood_value = - 1/2 * ( N*D * np.log( 2*np.pi*sigma2 ) + log_sum1 + 1/(sigma2) * ( log_sum2 - 2 * log_sum3 + log_sum4) )

    return np.sum(log_liklihood_value)

def global_search(x, M, limit):
    """
    Tries to find global maximum of the log-likelihood function.

    Parameters

    """
    best_log_likelihood = -1e10

    for iteration in range(0, limit):
        parameters = em(x, M, iteration+1)
        log_likelihood_value = log_likelihood(x, parameters)

        if log_likelihood_value > best_log_likelihood:
            best_log_likelihood = log_likelihood_value
            best_parameters = parameters

        if parameters[3] == 1000:
            print("EM algorithm did not converge within the limit of iterations")
            break

        printProgressBar(iteration, limit)

    return best_log_likelihood, best_parameters

"""
Be forewarned, I made this code my own. This means my comments are not
strictly professional. They are still informative and should get the
point across just fine. But I also make jokes, because it makes things
fun and reading the code less tedious.
"""

"""
!!! CHECK BEFORE EACH SCRIPT EXECUTION !!!
"""

snapshot = "N" # set the operating condition of the snapshot.   "N" for NOC and peruse fault_details.txt for the fault IDs
control = 1 # boolean, control (1) or no control (0) for the snapshot
autoreggressive_noise = 1
measurement_noise = 1
alpha = 0.05 # confidence level for the similarity checks

data = SimpleNamespace()
data.dt = None; # set the time difference manually (only needed if simulation time data not saved)

"""
Normal code execution below
"""
operating_conditons = pd.read_csv('fault_details.txt', sep=',')["Fault ID"].to_numpy()
operating_conditons = np.insert(operating_conditons, 0, "N")

## Snapshot data
data.snapshot = SimpleNamespace() # dedicated to snapshot-related values
with open(f"H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}/OldSnapshot_{snapshot}.json", 'r', encoding='utf-8') as file:
    data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
data.snapshot.fault = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
data.measured_variables = data.results["measurements"]

num = SimpleNamespace() # storing various useful numbers
num.recorded_points = len(data.results["t"])
# Getting the measurement time interval to slice out the window of the
# chosen operating condition (ignoring the NOC portion, if there is k
# disturbance)
data.t = data.results["t"]
data.dt = data.t[1] - data.t[0]

data.measured_variables.remove("t") # the measurement times are not being considered for PCA, removing it here saves effort later
num.variables = len(data.measured_variables); # the number of variables that will be in the matrix we perform PCA on

num.ignore = int(data.results["fault time"] / data.dt) # the number of measurements that will be ignored (corresponding to the period of NOC before the disturbance)
num.measurements = len(data.t) - num.ignore # the number of measurements that will be used for PCA (corresponding to the period after the disturbance)
data.set = np.zeros([num.recorded_points, num.variables]) # initialising the matrix that will store the dataset used to perform PCA

# putting all the simulation data into one matrix
for ii in range(0, num.variables):
    data.set[:, ii] = data.results[data.measured_variables[ii]]



# Bayesian PCA
data.snapshot.mean, data.snapshot.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
x = ((data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std).T # scaling the data
limit = 10
M = 9

# best_log_likelihood, best_parameters = global_search(x, M, limit)
# hinton(best_parameters[0])
# print("\n", best_log_likelihood)

# parameters = em(x, 14, 1, True)
# hinton(parameters[0])
# print(log_likelihood(x, parameters))
# print(parameters[0])


