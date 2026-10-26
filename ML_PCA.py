## Housekeeping
from types import SimpleNamespace
import json
import os
import subprocess
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

subprocess.run('cls', shell=True)
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
autoreggressive_noise = 0
measurement_noise = 1
alpha = 0.01 # confidence level for the similarity checks

data = SimpleNamespace()
data.dt = None; # set the time difference manually (only needed if simulation time data not saved)

"""
Normal code execution below
"""
operating_conditons = pd.read_csv('fault_details.txt', sep=',')["Fault ID"].to_numpy()
operating_conditons = np.insert(operating_conditons, 0, "N")

## Snapshot data
data.snapshot = SimpleNamespace() # dedicated to snapshot-related values
stats = SimpleNamespace()
with open(f"H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}/OldSnapshot_{snapshot}.json", 'r', encoding='utf-8') as file:
    data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
stats.snapshot_fault = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
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

# PCA
data.snapshot.mean, data.snapshot.std = np.mean(data.set, axis=0), np.std(data.set, axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
x = (data.set-data.snapshot.mean)/data.snapshot.std # scaling the data
U, S, Vt = np.linalg.svd(x, full_matrices=False) # computing the SVD of the scaled data
S_covar = S**2 / (num.measurements-1)
V, explained = np.transpose(Vt), 100 * S_covar / np.sum(S_covar) # converting the SVD results into the PCA stats
# choosing the number of principle components based on how many are needed
# to explain 95 % of the variance

cum_explained = [np.sum(explained[:ii]) for ii in range(0, num.variables)]
for ii in range(0, num.variables):
    if cum_explained[ii] > 95:
        k = ii + 1
        break
# k = num.variables
print(k)
# plt.figure(1)       
# plt.plot(np.linspace(1, num.variables, num.variables), cum_explained, color = 'k', linestyle = '-', label = "Cumulative explained variance")
# plt.plot([1, num.variables], [95, 95], color = 'r', linestyle = '--', label = "95% variance explained")
# plt.plot([k, k], [0, 100], color = 'b', linestyle = '--', label = f"Retained PCs = {k}")
# plt.xlabel("Number of principal components")
# plt.xticks(np.linspace(0, num.variables, num.variables+1))
# plt.ylabel("Cumulative explained variance (%)")
# plt.legend()
# plt.show(block=True)

# PCA stats for the chosen principal components
L = V[:, :k] # principal subspace of the snapshot
residual_eigenvals = S[k:] # eigenvalues
theta = [np.sum(residual_eigenvals**ii) for ii in range(1, 4)] # theta values
h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1]**2) # h0 value

# Similarity checks
planned_num_measurements = 1024
stats.tsquared95 = ( k * (planned_num_measurements - 1) / (planned_num_measurements - k) ) * f.ppf(1 - alpha, k, planned_num_measurements - k) # 95 % confidence limit for Hotelling's T-squared
stats.Q95 = theta[0] * ( norm.ppf(1-alpha) * h0 * np.sqrt( 2*theta[1] )/theta[0] + 1 + theta[1]*h0*(h0-1)/theta[0]**2 ) / h0 # 95 % confidence limit for the Q statistic
stats.r95 = binom.ppf(1-alpha, planned_num_measurements, alpha) # hardcoding the number of expected measurements... surely this will not bite me in the ass later...
stats.S_pcaBest = 0.965 # Johannesmeyer provided best value

covariance = S[:k]**2/(num.measurements-1) # covariance matrix

tsquared = np.zeros(num.measurements)
t = x@L
for row in range(0, num.measurements):
    tsquared[row] = t[row, :] @ np.diag( covariance**-1 ) @ t[row, :].T

# fig, ax = plt.subplots(1, 1)
# ax.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
# ax.plot(np.linspace(1, num.measurements, num.measurements), tsquared, color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
# # ax2[1].plot([num.ignore, num.ignore], [0, 30], color = 'r', linestyle = '-', label = "Fault occurrence")
# ax.set_xlabel("Observation")
# ax.set_ylabel("Hotelling's T-squared statistic")
# ax.set_title(f"Hotelling's T-squared statistic for snapshot")
# ax.legend(loc = 'upper left')
# fig.show()
# input("Press Enter to continue...")
# ax.cla()


## Historical data
start = 0
num.files = 1
while os.path.exists(f"H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}/run_{num.files}.json"):
    num.files = num.files + 1
num.files = num.files - 1 # the number of data files

start = 0
# num.files = 120
# Initialising arrays 
stats.tsquared_violations = np.zeros(num.files)
stats.tsquared_contributions = np.zeros([num.files, num.variables])
stats.Q_violations = np.zeros(num.files)
# stats.Q_cont = np.zeros([num.files, num.variables])
stats.S_pca = np.zeros(num.files)


data.H = SimpleNamespace()
stats.H_faults = np.empty(num.files, dtype=np.dtype('U5'))
# colours = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6', '#a9a9a9']
colours = ["#008080", "#A52A2A", "#FFD700", "#40E0D0", "#ADD8E6", "#000080", "#90EE90", "#006400", "#FFCCCB", "#8B0000", "#E6E6FA", "#4B0082", "#FFDAB9", "#D2691E", "#FFB6C1", "#C71585", "#D3D3D3", "#4A4A4A", "#000000", "#FFA500", "#8B8000"]
num.colours = len(colours)
markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "8", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]
# fig, ax = plt.subplots(2, 2)

for kk in range(start, num.files):
    # fig2, ax2 = plt.subplots(2, 1) # for T-squared contributions and T-squared statistic
    # fig3, ax3 = plt.subplots(1, 1) # for Q statistic
    with open(f"H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}/run_{kk+1}.json", 'r', encoding='utf-8') as file:
    # with open(f"H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}/run_{463}.json", 'r', encoding='utf-8') as file:
        data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
    stats.H_faults[kk] = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))

    if kk == start:
        num.recorded_points = len(data.results["t"])
        num.measurements = len(data.results["t"]) - num.ignore
        # num.measurements = len(data.results["t"])
        stats.tsquared = np.zeros([num.files, num.measurements])
        stats.Q = np.zeros([num.files, num.measurements])
         
        # y_med = np.median(explained)
        # print(f"y_med = {y_med}")
        # print(f"m={num.variables}")
        # print(f"n={num.measurements}")
        # # use the code supplement to calculate the coefficient, omega, using the above m x n matrix size
        # omega = 1.445137406658669
        # opt_hard_thres = omega*y_med
        # print(f"opt_hard_thres = {opt_hard_thres}")

        # for ii in range(0, num.variables):
        #     if explained[ii] < opt_hard_thres:
        #         k = ii
        #         break

        # plt.figure(1)       
        # plt.plot(np.linspace(1, num.variables, num.variables), explained, color = 'k', linestyle = '-', label = "Singular values")
        # plt.plot([1, num.variables], [opt_hard_thres, opt_hard_thres], color = 'r', linestyle = '--', label = "Optimal hard threshold")
        # plt.plot([k, k], [0, 100], color = 'b', linestyle = '--', label = f"Retained PCs = {k}")
        # plt.xlabel("Number of principal components")
        # plt.xticks(np.linspace(0, num.variables, num.variables+1))
        # plt.ylabel("Explained variance (%)")
        # plt.legend()
        # plt.show(block=True)
        

    data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    # putting all the simulation data into one matrix
    for ii in range(0,num.variables):
        data.set[:, ii] = data.results[data.measured_variables[ii]]

    # PCA                                                                                                               
    x = (data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std # scaling the data
    # x = (data.set[:, :]-data.snapshot.mean)/data.snapshot.std
    U, Snew, Vt = np.linalg.svd(x, full_matrices=False) # computing the SVD of the scaled data
    M = np.transpose(Vt[:k, :]) # shape(P) = (D, k), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
    # print(np.shape(x))
    
    # tsquared
    P = L
    t = x@P # score matrix
    # use of eigenvalues deprecated for the time being (since covar is not being used for the tsquared)
    # eigenvals = S[:k]**2 # eigenvalues

    # P = M
    # S_covar = Snew**2/(num.measurements-1) # covariance matrix
    
    for row in range(0, num.measurements):
        stats.tsquared[kk, row] = t[row, :] @ np.diag( S_covar[:k]**-1 ) @ t[row, :].T

    for variable in range(0, num.variables):
        for measurement in range(0, num.measurements):
            contribution = t[measurement, :] @ np.diag( S_covar[:k]**-1 ) @ P[variable, :] * x[measurement,variable]
            stats.tsquared_contributions[kk, variable] += contribution
            # stats.tsquared[kk, measurement] += contribution

    for tsquared in stats.tsquared[kk]:
        stats.tsquared_violations[kk] += tsquared > stats.tsquared95
    
    # stats.tsquared_contributions[kk, :] = np.sum([[np.sum([ 0 if (t[row, component] / S[component])**2 < stats.tsquared95**-k else t[row, component] / S[component]**2 * P[variable,component] * x[row,variable] for component in range(0, k)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

    # Q statistic
    xbar_scaled = t@P.T # shape(t) = (n, m), projection of the the score matrix back onto the original variable space
    e = x - xbar_scaled # shape(t) = (n, m), residuals or error, inter alia
    stats.Q[kk, :] = np.sum(e**2, axis=1)
    for Q in stats.Q[kk, :]:
        stats.Q_violations[kk] += Q > stats.Q95
    # Q contributions not really meaningful
    # stats.Q_cont[kk, :] = np.sum([[r[row, variable] * r[row, variable] for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

    # PCA similarity statistic
    # M = P
    stats.S_pca[kk] = np.trace(L.T@M@M.T@L) / k  

    fault_index = np.where(operating_conditons == stats.H_faults[kk])[0][0]

    # # plt.figure(1) # tsquared contributions
    # ax[0, 0].plot(data.measured_variables, stats.tsquared_contributions[kk, :], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='-', linewidth=0.3, label=f'{kk}, {stats.H_faults[kk]}')

    # # plt.figure(2) # PC 1 versus 2
    # # plt.axes(projection='3d')
    # ax[0, 1].plot(t[:, 0], t[:, 1], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

    # # plt.figure(3) # PC 2 versus 3
    # ax[1, 0].plot(t[:, 1], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

    # # plt.figure(4) # PC 1 versus 3
    # ax[1, 1].plot(t[:, 0], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

    # ax2[0].bar(data.measured_variables, stats.tsquared_contributions[kk, :], color=colours[fault_index % num.colours], linestyle='-', linewidth=0.3, label=f'{kk}, {stats.H_faults[kk]}')
    # ax2[1].plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
    # ax2[1].plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared[kk, :], color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
    # ax2[1].plot([num.ignore, num.ignore], [0, 30], color = 'r', linestyle = '-', label = "Fault occurrence")
    # ax2[1].set_xlabel("Observation")
    # ax2[1].set_ylabel("Hotelling's T-squared statistic")
    # ax2[1].set_title(f"Hotelling's T-squared statistic for run {kk+1}, {stats.H_faults[kk]}")
    # ax2[1].legend(loc = 'upper left')
    # ax2[0].set_title(f"Hotelling's T-squared contributions from each variable")
    # fig2.show()
    # print(f"Violations: {stats.tsquared_violations[kk]} out of {stats.r95}")
    # input("Press Enter to continue...")
    # ax2[0].cla()
    # ax2[1].cla()

    # ax3.plot(np.linspace(1, num.measurements, num.measurements), stats.Q95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
    # ax3.plot(np.linspace(1, num.measurements, num.measurements), stats.Q, color = 'k', linestyle = '-', label = "Q statistic")
    # ax3.plot([num.ignore, num.ignore], [0, 30], color = 'r', linestyle = '-', label = "Fault occurrence")
    # ax3.set_xlabel("Observation")
    # ax3.set_ylabel("Q statistic (Mean Squared Prediction Error)")
    # ax3.set_title(f"Q statistic for run {kk+1}, {stats.H_faults[kk]}")
    # ax3.legend(loc = 'upper left')
    # fig3.show()
    # print(f"Violations: {stats.Q_violations[kk]} out of {stats.r95}")
    # input("Press Enter to continue...")
    # ax3.cla()

    printProgressBar(kk+1, num.files)

names = list(stats.__dict__.keys())
save = {}
for index, item in enumerate(stats.__dict__.values()):
    if type(item) == np.ndarray:
        save[names[index]] = item.tolist() 
    else:
        save[names[index]] = item
# with open("results/PCA(cum_explained).json", "w") as json_file:
#     json.dump(save, json_file)
# with open("results/PCA(OHT).json", "w") as json_file:
#     json.dump(save, json_file)


## Similarity stats
# A priori labelling for evaluating performance, not available for use
matches = stats.H_faults == stats.snapshot_fault
print(f"matches = {matches}")
not_matches = [int(ii) for ii in ~matches.astype(bool)]

# Quantifying performance
Np = np.zeros(3)
N1 = np.zeros(3)
N2 = np.zeros(3)
p = np.zeros(3)
n = np.zeros(3)
nmax = np.zeros(3)
e = np.zeros(3)
Ndb = np.sum(matches) # historical windows that are actually similar

print(f"Snapshot fault: {stats.snapshot_fault}\n")

for ii, statistic in enumerate([["tsquared_violations", stats.r95],["Q_violations", stats.r95]]):#, ["Q_violations", stats.r95]]):
    candidate_pool = [int(ii) for ii in getattr(stats, statistic[0]) < statistic[1]]
    Np[ii] = np.sum(getattr(stats, statistic[0]) < statistic[1]) # candidate pool size (number of data sets classified as similar)
    N1[ii] = np.sum((getattr(stats, statistic[0]) < statistic[1]) * (matches)) # candidate pool data sets that are actually similar
    N2[ii] = np.sum((getattr(stats, statistic[0]) < statistic[1]) * (not_matches)) # candidate pool data sets that are not similar (N2 = Np - N1)
    p[ii] = N1[ii] / Np[ii] * 100
    n[ii]= N1[ii] / Ndb * 100
    nmax[ii] = np.min([Np[ii]/Ndb, 1]) * 100
    e[ii] = ( p[ii] + n[ii] ) / 2

    print(f"\n{statistic[0]} Results:")
    print(f"Np     {Np[ii]}\nN1     {N1[ii]}\nN2     {N2[ii]}\nNdb    {Ndb}\np      {p[ii]}\nn      {n[ii]}\nnmax   {nmax[ii]}\ne      {e[ii]}")


# stats.S_pcaBest = 0.965
ii=2
candidate_pool = [int(ii) for ii in getattr(stats, "S_pca") > stats.S_pcaBest]
Np[ii] = np.sum(stats.S_pca > stats.S_pcaBest) # candidate pool size (number of data sets classified as similar)
N1[ii] = np.sum((stats.S_pca > stats.S_pcaBest) * (matches)) # candidate pool data sets that are actually similar
N2[ii] = np.sum((stats.S_pca > stats.S_pcaBest) * (not_matches)) # candidate pool data sets that are not similar (N2 = Np - N1)
p[ii] = N1[ii] / Np[ii] * 100
n[ii]= N1[ii] / Ndb * 100
nmax[ii] = np.min([Np[ii]/Ndb, 1]) * 100
e[ii] = ( p[ii] + n[ii] ) / 2

print(f"\nS_pca Results:")
print(f"Np     {Np[ii]}\nN1     {N1[ii]}\nN2     {N2[ii]}\nNdb    {Ndb}\np      {p[ii]}\nn      {n[ii]}\nnmax   {nmax[ii]}\ne      {e[ii]}")

legend_elements = [Line2D([0], [0], marker = markers[np.where(operating_conditons == fault)[0][0]], color = colours[np.where(operating_conditons == fault)[0][0] % num.colours], label=fault, linestyle='None', markerfacecolor='none') for fault in operating_conditons]



# # plt.figure(1)
# fig.legend(handles=legend_elements, loc = 'right')
# ax[0, 0].set_ylabel("Hotelling's T-squared contributions")
# ax[0, 0].set_xlabel("Measured variables")
# # plt.figure(2)
# # plt.legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor = (1, 1))
# ax[0, 1].set_xlabel("PC 1")
# ax[0, 1].set_ylabel("PC 2")
# # plt.figure(3)
# # plt.legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor = (1, 1))
# ax[1, 0].set_xlabel("PC 2")
# ax[1, 0].set_ylabel("PC 3")
# # plt.figure(4)
# # plt.legend(handles=legend_elements, loc = 'upper left', bbox_to_anchor = (1, 1))
# ax[1, 1].set_xlabel("PC 1")
# ax[1, 1].set_ylabel("PC 3")


print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")

# plt.show()
# input("Press Enter to continue...")

