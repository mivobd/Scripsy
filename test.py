## Housekeeping
from types import SimpleNamespace
import json
import subprocess
import os
import numpy as np
from scipy.stats import f, norm, binom, chi2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time
import shutil
from pathlib import Path

subprocess.run('cls', shell=True)

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

control = 1
autoreggressive_noise = 0
measurement_noise = 1

folder_path = f"BPCA_H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}"
results = SimpleNamespace()
num = SimpleNamespace()

with open(f"{folder_path}/sigma2_test_463.json", 'r', encoding='utf-8') as file:
    results.H = SimpleNamespace(**json.load(file))

W_unordered = np.array(results.H.best_parameters[0])
alphas = 1/np.array(results.H.best_parameters[2])
num.OGweights = len(alphas)
num.variables = np.shape(W_unordered)[0]
ordered_indices = np.argsort(alphas[:, 0])
W = W_unordered[:, ordered_indices[::-1]]
alphas_ordered = alphas[ordered_indices[::-1]]
sigma2 = results.H.best_parameters[1][0][0]

cum_explained = [np.sum(alphas_ordered[:ii+1]) for ii in range(0, num.OGweights)]/np.sum(alphas_ordered)
for ii in range(0, num.OGweights):
    if cum_explained[ii] > 0.95:
        k = ii+1
        break
# print(num.OGweights, k)
retained = np.zeros((num.variables, num.OGweights))
for ii in range(0, k):
    retained[:, ii] = W[:, ii].reshape(14)

# comparing original BPCA results to ordered and retained components
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
hinton(W_unordered, ax1)
ax1.set_title("Unordered")
ax1.set_xlabel("Features")
ax1.set_ylabel("Directions")
hinton(W, ax2)
ax2.set_title("Ordered")
ax2.set_xlabel("Features")
hinton(retained, ax3)
ax3.set_title("Retained")
ax3.set_xlabel("Features")
plt.show()