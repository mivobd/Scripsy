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

"""
!!! CHECK BEFORE EACH SCRIPT EXECUTION !!!
"""

true_M = 3
lim = 0.05

"""
Normal code execution below
"""

num = SimpleNamespace()
stats = SimpleNamespace()


with open("simulation_data/test_x.json", "r") as json_file:
    x = np.array(json.load(json_file))

D, N = np.shape(x)
num.variables = D
num.measurements = N

validation = SimpleNamespace()
with open(f"BPCA_models/MATLAB_PPCA.json", 'r', encoding='utf-8') as file:
    validation.results = json.load(file)

k = int(validation.results["k"])
validation.W = np.reshape(np.array(validation.results["W"]), (D, k))
validation.beta = 1/validation.results["v"]

I_K = np.identity(k)
I_D = np.identity(num.variables)

# x = x.T
P = validation.W # principal subspace
M = P.T @ P + I_K / validation.beta
S_var_tx_inv = validation.beta*M
validation.t = np.linalg.solve( M, P.T ) @ x # score matrix for BPCA
tsquared95 = chi2.ppf(1-lim, k) # Bayesian PCA version of 95 % confidence limit for Hotelling's T-squared 

validation.tsquared = np.zeros([num.measurements])
for column in range(0, num.measurements):
    validation.tsquared[column] = (validation.t[:, column].reshape(-1, 1).T @ S_var_tx_inv @ validation.t[:, column])[0]

results = SimpleNamespace()
with open(f"BPCA_models/test_results.json", 'r', encoding='utf-8') as file:
    results.Snapshot = SimpleNamespace(**json.load(file))

W = np.array(results.Snapshot.best_parameters[0])
plt.show()
beta = results.Snapshot.best_parameters[1]

I_K = np.identity(9)
I_D = np.identity(num.variables)

P = W # principal subspace
M = P.T @ P + I_K / beta
S_var_tx_inv = beta*M
t = np.linalg.solve( M, P.T ) @ x # score matrix for BPCA

stats.tsquared = np.zeros([num.measurements])
for column in range(0, num.measurements):
    stats.tsquared[column] = (t[:, column].reshape(-1, 1).T @ S_var_tx_inv @ t[:, column])[0]

fig, ax = plt.subplots(1, 1, layout = "constrained") # for T-squared contributions and T-squared statistic
ax.plot(np.linspace(1, num.measurements, num.measurements), tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
ax.plot(np.linspace(1, num.measurements, num.measurements), validation.tsquared, color = 'k', linestyle = '-', label = "MATLAB PPCA")
ax.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared, color = 'g', linestyle = ':', label = "BPCA EM without ARD")
ax.set_xlabel("Observation")
ax.set_ylabel("Hotelling's T-squared statistic")
ax.legend(loc = 'upper left')

fig2, ax2 = plt.subplots(1, 2, layout = "constrained")
hinton(validation.W, ax = ax2[0])
ax2[0].set_title("MATLAB PPCA")
ax2[0].set_xlabel("Features")
ax2[0].set_ylabel("Dimensions")
hinton(W, ax = ax2[1])
ax2[1].set_title("BPCA EM without ARD")
ax2[1].set_xlabel("Features")
ax2[1].set_ylabel("Dimensions")

print(f"MATLAB beta: {validation.beta}")
print(f"BPCA EM without ADR beta: {beta}")

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")
plt.show()
