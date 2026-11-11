## Housekeeping
from types import SimpleNamespace
import json
import os
import subprocess
import numpy as np
from scipy.stats import f, norm, binom
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import time

subprocess.run('cls', shell=True)
timer_start = time.perf_counter()
plt.rcParams.update({'font.size': 16})


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

def plot_results(results, fig, ax):
    num_recorded_points = len(results["t"])
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = ax

    # to get valve level action
    system = signal.TransferFunction([1/16], [2/60, 1])

    # level
    ax1.plot(results["t"], results["h"], label="measurement", color="blue")
    ax1.plot(results["t"], 6 * np.ones(num_recorded_points), label="setpoint", color="orange")
    ax1.set_ylim([5, 7])
    ax1.set_ylabel("Level [m]")
    ax1.set_xlim([0, results["t"][-1]])
    ax1.set_xlabel("Time [min]")

    # outlet flow
    ax2.plot(results["t"], results["Q"], color="blue")
    ax2.plot(results["t"], 100 + np.array(results["hC"]), color="orange")
    ax2.set_ylim([85, 115])
    ax2.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax2.set_xlim([0, results["t"][-1]])
    ax2.set_xlabel("Time [min]")

    # outlet valve
    t_out, l, x_out = signal.lsim(system, U=results["QC"], T=results["t"])
    ax3.plot(results["t"], l * 100 + 50, color="blue")
    ax3.set_ylim([0, 100])
    ax3.set_ylabel("Opening [%]")
    ax3.set_xlim([0, results["t"][-1]])
    ax3.set_xlabel("Time [min]")

    # temperature
    ax4.plot(results["t"], results["T"], color="blue")
    ax4.plot(results["t"], 402.35 * np.ones(num_recorded_points), color="orange")
    ax4.set_ylim([382.35, 422.35])
    ax4.set_ylabel("Temperature [K]")
    ax4.set_xlim([0, results["t"][-1]])
    ax4.set_xlabel("Time [min]")

    # coolant flow
    ax5.plot(results["t"], results["Q_C"], color="blue")
    ax5.plot(results["t"], 15 + np.array(results["TC"]), color="orange")
    ax5.set_ylim([0, 30])
    ax5.set_ylabel(r"Flow rate [L min$^{-1}$]")
    ax5.set_xlim([0, results["t"][-1]])
    ax5.set_xlabel("Time [min]")

    # coolant valve
    t_out, l_C, x_out = signal.lsim(system, U=results["Q_CC"], T=results["t"])
    ax6.plot(results["t"], l_C * 100 + 50, color="blue")
    ax6.set_ylim([0, 100])
    ax6.set_ylabel("Opening [%]")
    ax6.set_xlim([0, results["t"][-1]])
    ax6.set_xlabel("Time [min]")

    fig.legend(loc='outside upper center', ncols=2)

    return None


"""
!!! CHECK BEFORE EACH SCRIPT EXECUTION !!!
"""

snapshot = "N" # string, set the operating condition of the snapshot.   "N" for NOC and peruse fault_details.txt for the fault IDs. Classifying faults did not end up being part of the research, so this will never change.
control = 1 # integer boolean, control (1) or no control (0) in the snapshot. Choice reflected in digit following "C" in dataset filename.
autoregressive_noise = 1 # integer boolean, as above, but with autoregressive noise. Choice reflected in digit following "AR" in dataset filename.
measurement_noise = 1 # integer boolean, I think you get the point. Choice reflected in digit following "M" in dataset filename.

component_selection = 1 # integer boolean, 0 for cumulative explained variance, 1 for optimal hard threshold, 2 for Scree plot (needs input). Choice reflected in results filename.
lim = 0.05 # float, confidence level for the similarity checks, I checked 0.05 and 0.01 in this research. Choice reflected in results filename.

plot_PC_selection = False # boolean, set to true to view how the number of PCs was chosen
show_snapshot_Tsquared = False # boolean, set to true to view the T-squared statistic results for the snapshot data
show_disturbance_effect = False # boolean, set to true to view the effect on the system of the disturbance (also shows the snapshot's data)

plot_statistics = False # boolean, set to true to view T-squared and Q-statistic plots for each run (highly recommend using the "view_specific_run" to check random runs. Most interesting are step change faults.
check_contributions = False # boolean, prints the value of T-squared calculated normally versus as the sum of contributions from variables.
view_specific_run = 98 # integer, "None" by default for normal script execution. Set to the run number you want to view.
plot_components = False # boolean, set to true to view the first three principal components for each run on a 3D scatter plot (highly recommend using the "view_specific_run" to check random runs. Most interesting are step change faults.
poster_plots = False # boolean, set to true to view the plots in a format suitable for poster presentation.


"""
Normal code execution below
"""
operating_conditons = pd.read_csv('fault_details.txt', sep=',')["Fault ID"].to_numpy() # all the possible faults and related information. Consult Johannesmeyer et al. (2002) or table 5 in the project report.
operating_conditons = np.insert(operating_conditons, 0, "N") # adding normal operation to the list

## Snapshot data
data = SimpleNamespace()
data.snapshot = SimpleNamespace() # dedicated to snapshot-related values
stats = SimpleNamespace()
simulation_path = f"simulation_data/C{control}_AR{autoregressive_noise}_M{measurement_noise}"
with open(f"{simulation_path}/OldSnapshot_{snapshot}.json", 'r', encoding='utf-8') as file:
    data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
stats.snapshot_fault = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
data.measured_variables = data.results["measurements"]

if show_disturbance_effect == True:
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), layout='constrained', num='Snapshot data')
    plot_results(data.results, fig, ax)
    plt.show()

num = SimpleNamespace() # storing various useful numbers
num.recorded_points = len(data.results["t"])
# Getting the measurement time interval to slice out the window of the
# chosen operating condition (ignoring the NOC portion, if there is a
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

if component_selection == 0:

    selection_type = "CEV"

    # choosing the number of principle components based on how many are needed to explain 95 % of the variance
    stop = False
    cum_explained = np.zeros(num.variables)
    for ii in range(0, num.variables):
        if ii == 0:
            cum_explained[ii] += explained[ii]
        else:
            cum_explained[ii] += cum_explained[ii-1] + explained[ii]

        if cum_explained[ii] > 95 and stop == False:
            k = ii + 1
            stop = True

    if plot_PC_selection == True:
        plt.figure(1, layout='constrained') 
        plt.plot(np.linspace(1, num.variables, num.variables), cum_explained, color = 'k', linestyle = '-', label = "Cumulative explained variance")
        plt.plot([1, num.variables], [95, 95], color = 'r', linestyle = '--', label = "95% variance explained")
        plt.plot([k, k], [0, 100], color = 'g', linestyle = '--', label = f"Retained PCs = {k}")
        plt.xlabel("Number of principal components")
        plt.xticks(np.linspace(0, num.variables, num.variables+1))
        plt.ylabel("Cumulative explained variance (%)")
        plt.legend()
        plt.show()

elif component_selection == 1:
    
    selection_type = "OHT"

    # choosing the PCs retained using an optimal hard threshold
    y_med = np.median(S)
    print(f"y_med = {y_med}")
    print(f"m={num.variables}")
    print(f"n={num.measurements}")
    # use the code supplement to calculate the coefficient, omega, using the above m x n matrix size
    omega = 1.416496421203151
    opt_hard_thres = omega*y_med
    print(f"opt_hard_thres = {opt_hard_thres}")
    for ii in range(0, num.variables):
        if S[ii] < opt_hard_thres:
            k = ii
            break

    if plot_PC_selection == True:
        plt.figure(1, layout='constrained')       
        plt.bar(np.linspace(1, num.variables, num.variables), S, color = 'b', linestyle = '-')
        plt.plot(np.linspace(1, num.variables, num.variables), S, color = 'k', linestyle = '--', label = "Singular values")
        plt.plot([1, num.variables], [opt_hard_thres, opt_hard_thres], color = 'r', linestyle = '--', label = "Optimal hard threshold")
        plt.plot([k, k], [0, np.max(S)], color = 'g', linestyle = '--', label = f"Retained PCs = {k}")
        plt.xlabel("Corresponding principal components")
        plt.xticks(np.linspace(0, num.variables, num.variables+1))
        plt.ylabel("Singular values")
        plt.legend()
        plt.show()

elif component_selection == 2:
    selection_type = "Scree"

    plt.figure(1, layout='constrained')       
    plt.bar(np.linspace(1, num.variables, num.variables), S_covar, color = 'b', linestyle = '-')
    plt.plot(np.linspace(1, num.variables, num.variables), S_covar, color = 'k', linestyle = '--')
    plt.xlabel("Corresponding principal components")
    plt.xticks(np.linspace(0, num.variables, num.variables+1))
    plt.ylabel("Eigenvalues")
    plt.show()
    k = int(input("Enter the number of principal components to retain: "))

print(f"{k} PCs retained\n")

## PCA statistics for the chosen principal components
x = x.T
L = V[:, :k] # principal subspace of the snapshot
residual_eigenvals = S[k:]**2/(num.measurements-1) # eigenvalues corresponding to the residual subspace
theta = [np.sum(residual_eigenvals**ii) for ii in range(1, 4)] # theta values
h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1]**2) # h0 value

## Monitoring statistic confidence limits
planned_num_measurements = 1024 # hardcoding the number of expected measurements... surely this will not bite me in the ass later... if this is ever used by someone else I guarantee this will be a reason for the code breaking. In general I have hardcoded so many expected save formats. Tangent over, enjoy reading on (I hope).
stats.tsquared95 = ( k * (planned_num_measurements - 1) / (planned_num_measurements - k) ) * f.ppf(1 - lim, k, planned_num_measurements - k) # 95 % confidence limit for Hotelling's T-squared
stats.Q95 = theta[0] * ( norm.ppf(1-lim) * h0 * np.sqrt( 2*theta[1] )/theta[0] + 1 + theta[1]*h0*(h0-1)/theta[0]**2 ) / h0 # 95 % confidence limit for the Q statistic
stats.r95 = binom.ppf(1-lim, planned_num_measurements, lim) 
stats.S_pcaBest = 0.965 # Johannesmeyer provided best value


t = L.T @ x
if show_snapshot_Tsquared == True:
    tsquared = np.zeros(num.measurements)

    for column in range(0, num.measurements):
        tsquared[column] = t[:, column].T @ np.diag( S_covar[:k]**-1 ) @ t[:, column]

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
    ax.plot(np.linspace(1, num.measurements, num.measurements), tsquared, color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
    ax.set_xlabel("Observation")
    ax.set_ylabel("Hotelling's T-squared statistic")
    ax.set_title(f"Hotelling's T-squared statistic for snapshot")
    ax.legend(loc = 'upper left')
    plt.show()

## Historical data
if view_specific_run != None:
    start = view_specific_run - 1
    num.files = view_specific_run
else:
    start = 0
    num.files = 1
    while os.path.exists(f"{simulation_path}/run_{num.files}.json"):
        num.files = num.files + 1
    num.files = num.files - 1 # the number of data files

## Initialising variables for storing statistics
stats.tsquared_violations = np.zeros(num.files)
stats.tsquared_contributions = np.zeros([num.files, num.variables])
stats.Q_violations = np.zeros(num.files)
stats.Q_contributions = np.zeros([num.files, num.variables])
stats.S_pca = np.zeros(num.files)

## Initialising variables for storing historical snapshot data and fault types
data.H = SimpleNamespace()
stats.H_faults = np.empty(num.files, dtype=np.dtype('U5'))


## Setup for plots, different colour and marker for each operating condition.
colours = ["#008080", "#A52A2A", "#FFD700", "#40E0D0", "#ADD8E6", "#000080", "#90EE90", "#006400", "#FFCCCB", "#8B0000", "#E6E6FA", "#4B0082", "#FFDAB9", "#D2691E", "#FFB6C1", "#C71585", "#D3D3D3", "#4A4A4A", "#000000", "#FFA500", "#8B8000"]
num.colours = len(colours)
markers = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "p", "8", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]


if plot_components == True:
    fault_index = 0
    fig3D = plt.figure()
    ax3D = fig3D.add_subplot(projection='3d')
    ax3D.scatter(t[:, 0], t[:, 1], t[:, 2],  marker = markers[fault_index], color=colours[fault_index % num.colours], linestyle='None', label=f'NOC')

    fig, ax = plt.subplots(1, 3, layout='constrained')
    # PC 1 versus 2
    ax[0].plot(t[:, 0], t[:, 1], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'Snapshot')
    ax[0].set_xlabel("PC 1")
    ax[0].set_ylabel("PC 2")

    # PC 2 versus 3
    ax[1].plot(t[:, 1], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'Snapshot')
    ax[1].set_xlabel("PC 2")
    ax[1].set_ylabel("PC 3")

    # PC 1 versus 3
    ax[2].plot(t[:, 0], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'Snapshot')
    ax[2].set_xlabel("PC 1")
    ax[2].set_ylabel("PC 3")

# start = 97
for kk in range(start, num.files):
    fig2, ax2 = plt.subplots(2, 1, layout='constrained') # for T-squared contributions and T-squared statistic
    fig3, ax3 = plt.subplots(2, 1, layout='constrained') # for Q statistic

    with open(f"simulation_data/C{control}_AR{autoregressive_noise}_M{measurement_noise}/run_{kk+1}.json", 'r', encoding='utf-8') as file:
        data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
    stats.H_faults[kk] = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
    fault_index = np.where(operating_conditons == stats.H_faults[kk])[0][0] # used for plotting, pairs fault to corresponding colour and marker

    if show_disturbance_effect == True:
        fig_dist, ax_dist = plt.subplots(2, 3, figsize=(12, 8), layout='constrained', num='Historical data')
        plot_results(data.results, fig_dist, ax_dist)
        fig_dist.show()

    if kk == start:
        num.recorded_points = len(data.results["t"])

        if plot_statistics == True:
            num.measurements = num.recorded_points
        else:
            num.measurements = len(data.results["t"]) - num.ignore

        stats.tsquared = np.zeros([num.files, num.measurements])
        stats.Q = np.zeros([num.files, num.measurements])

        def observation_to_time(observation):
            return observation * data.dt
        def time_to_observation(time):
            return time / data.dt
        

    data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    # putting all the simulation data into one matrix
    for ii in range(0,num.variables):
        data.set[:, ii] = data.results[data.measured_variables[ii]]

    # PCA                                                                                                               
    if plot_statistics == True:
        x = (data.set[:, :]-data.snapshot.mean)/data.snapshot.std
    else:
        x = (data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std # scaling the data
    U, Snew, Vt = np.linalg.svd(x, full_matrices=False) # computing the SVD of the scaled data
    M = np.transpose(Vt[:k, :]) # shape(P) = (D, k), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
    
    # retained PC subspace projections
    x = x.T # data was stored in row based convention, transposed to suit the convention of the report
    P = L
    t = P.T @ x # score matrix

    ## Hotelling's T-squared statistic calculations
    for column in range(0, num.measurements):
        stats.tsquared[kk, column] = t[:, column].T @ np.diag( S_covar[:k]**-1 ) @ t[:, column]

    if check_contributions == True:
        print(f"T-squared statistic calculated normally: {stats.tsquared[kk, :]}")
        stats.tsquared[kk, :] = 0
    for variable in range(0, num.variables):
        for measurement in range(0, num.measurements):
            
            contribution = P[variable, :].T @ np.diag( S_covar[:k]**-1 ) @ t[:, measurement].T * x[variable, measurement]
            stats.tsquared_contributions[kk, variable] += contribution

            if check_contributions == True:
                stats.tsquared[kk, measurement] += contribution

    if check_contributions == True:
        print(f"T-squared statistic calculated by summing contributions: {stats.tsquared[kk, :]}")

    for tsquared in stats.tsquared[kk]:
        stats.tsquared_violations[kk] += tsquared > stats.tsquared95
    
    
    ## Q-statistic calculations
    xbar = P @ t # shape(t) = (n, m), projection of the the score matrix back onto the original variable space
    e = x - xbar # shape(t) = (n, m), residuals or error, inter alia
    stats.Q[kk, :] = np.sum(e**2, axis=0)

    if check_contributions == True:
        print(f"Q-statistic calculated normally: {stats.Q[kk, :]}")
        stats.Q[kk, :] = 0
    for column in range(0, num.measurements):
        for variable in range(0, num.variables):
            contribution = e[variable, column] * e[variable, column]
            stats.Q_contributions[kk, variable] += contribution

            if check_contributions == True:
                stats.Q[kk, column] += contribution

    if check_contributions == True:
            print(f"Q-statistic calculated by summing contributions: {stats.Q[kk, :]}")
            input("Press Enter to continue...")

    stats.Q[kk, :] = np.sum(e**2, axis=0)

    for Q in stats.Q[kk, :]:
        stats.Q_violations[kk] += Q > stats.Q95

    ## PCA similarity statistic calculations
    stats.S_pca[kk] = np.trace( L.T @ M @ M.T @ L ) / k  

    if plot_components == True:
        # PC 1 versus 2
        ax[0].plot(t[:, 0], t[:, 1], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

        # PC 2 versus 3
        ax[1].plot(t[:, 1], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

        # PC 1 versus 3
        ax[2].plot(t[:, 0], t[:, 2], marker = markers[fault_index], markerfacecolor='none', color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

        ax3D.scatter(t[:, 0], t[:, 1], t[:, 2], marker = markers[fault_index], color=colours[fault_index % num.colours], linestyle='None', label=f'{kk}, {stats.H_faults[kk]}')

    if plot_statistics == True:
        ax2[0].bar(data.measured_variables, stats.tsquared_contributions[kk, :], color=colours[fault_index % num.colours], linestyle='-', linewidth=0.3, label=f'{kk}, {stats.H_faults[kk]}')
        ax2[1].plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
        ax2[1].plot(np.linspace(1, num.measurements, num.measurements), stats.tsquared[kk, :], color = 'k', linestyle = '-', label = "Hotelling's T-squared statistic")
        ax2[1].plot([num.ignore, num.ignore], [0, max(max(stats.tsquared[kk, :]),np.max(stats.tsquared95))], color = 'b', linestyle = '-.', label = "Fault occurrence")
        ax2[1].xaxis.tick_top()
        ax2[1].xaxis.set_label_position('top')
        ax2[1].set_xlabel("Observation")
        ax2[1].set_ylabel("Hotelling's T-squared statistic")
        ax2[1].set_title(f"Hotelling's T-squared statistic for run {kk+1}, {stats.H_faults[kk]}")
        ax2[1].legend(loc = 'upper left')
        ax2[0].set_title(f"Hotelling's T-squared contributions from each variable")
        secax2 = ax2[1].secondary_xaxis('bottom', functions=(observation_to_time, time_to_observation))
        secax2.set_xlabel('Time (min)')
        fig2.show()

        ax3[0].bar(data.measured_variables, stats.Q_contributions[kk, :], color=colours[fault_index % num.colours], linestyle='-', linewidth=0.3, label=f'{kk}, {stats.H_faults[kk]}')
        ax3[1].plot(np.linspace(1, num.measurements, num.measurements), stats.Q95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
        ax3[1].plot(np.linspace(1, num.measurements, num.measurements), stats.Q[kk, :], color = 'k', linestyle = '-', label = "Q statistic")
        ax3[1].plot([num.ignore, num.ignore], [0, max(max(stats.Q[kk, :]),np.max(stats.Q95))], color = 'b', linestyle = '-.', label = "Fault occurrence")
        ax3[1].set_xlabel("Observation")
        ax3[1].xaxis.tick_top()
        ax3[1].xaxis.set_label_position('top')
        ax3[1].set_ylabel("Q statistic (Squared Prediction Error)")
        ax3[1].set_title(f"Q statistic for run {kk+1}, {stats.H_faults[kk]}")
        ax3[1].legend(loc = 'upper left')
        ax3[0].set_title(f"Q-statistic (SPE) contributions from each variable")
        secax3 = ax3[1].secondary_xaxis('bottom', functions=(observation_to_time, time_to_observation))
        secax3.set_xlabel('Time (min)')
        fig3.show()


        input("Press Enter to continue...")
        plt.close(fig2)
        plt.close(fig3)

    if poster_plots == True:
        ax2[0].plot(np.linspace(5/60, 120, num.measurements), stats.tsquared95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence")
        ax2[0].plot(np.linspace(5/60, 120, num.measurements), stats.tsquared[kk, :], color = 'k', linestyle = '-')
        ax2[0].plot([num.ignore*5/60, num.ignore*5/60], [0, max(max(stats.tsquared[kk, :]),np.max(stats.tsquared95))], color = 'b', linestyle = '-.', label = "Fault occurrence")
        ax2[0].set_ylabel(r"T$^{2}$ statistic", fontsize=30)
        ax2[0].legend(loc = 'upper left', fontsize=30)
        ax2[0].tick_params(axis='both', labelsize=26)

        ax2[1].plot(np.linspace(5/60, 120, num.measurements), stats.Q95*np.ones(num.measurements), color = 'r', linestyle = '--', label = "95% confidence limit")
        ax2[1].plot(np.linspace(5/60, 120, num.measurements), stats.Q[kk, :], color = 'k', linestyle = '-', label = "Q statistic")
        ax2[1].plot([num.ignore*5/60, num.ignore*5/60], [0, max(max(stats.Q[kk, :]),np.max(stats.Q95))], color = 'b', linestyle = '-.', label = "Fault occurrence")
        ax2[1].set_xlabel("Time (min)", fontsize=30)
        ax2[1].set_ylabel("Q-statistic", fontsize=30)
        ax2[1].tick_params(axis='both', labelsize=26)

        ax2[0].set_title(f"Fault: Inlet Flow Rate Step Change", fontsize=36)
        fig2.show()

    elif view_specific_run != None:
        print(f"T-squared violations: {stats.tsquared_violations[kk]} out of {stats.r95}")
        print(f"Q violations: {stats.Q_violations[kk]} out of {stats.r95}")
        input("Press Enter to continue...")

    printProgressBar(kk+1, num.files)

if view_specific_run == None:
    ## saving results
    names = list(stats.__dict__.keys())
    save = {}
    for index, item in enumerate(stats.__dict__.values()):
        if type(item) == np.ndarray:
            save[names[index]] = item.tolist() 
        else:
            save[names[index]] = item
    with open(f"results/PCA({selection_type})_C{control}_AR{autoregressive_noise}_M{measurement_noise}_{int(100*(1-lim))}a.json", "w") as json_file:
        json.dump(save, json_file)

if plot_components == True:
    legend_elements = [Line2D([0], [0], marker = markers[np.where(operating_conditons == fault)[0][0]], color = colours[np.where(operating_conditons == fault)[0][0] % num.colours], label=fault, linestyle='None', markerfacecolor='none') for fault in operating_conditons]
    
    ax[0].set_xlabel("PC 1")
    ax[0].set_ylabel("PC 2")
    ax[1].set_xlabel("PC 2")
    ax[1].set_ylabel("PC 3")
    ax[2].set_xlabel("PC 1")
    ax[2].set_ylabel("PC 3")
    ax[2].legend(handles=legend_elements, loc = 'right')

    ax3D.set_xlabel("PC 1")
    ax3D.set_ylabel("PC 2")
    ax3D.set_zlabel("PC 3")
    ax3D.legend(handles=legend_elements, loc = 'center left')
    plt.show()

print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")


## References
# Johannesmeyer, M.C., Singhal, A. & Seborg, D.E. 2002. Pattern matching in historical data. AIChE Journal. 48(9):2022–2038. DOI: 10.1002/aic.690480916.