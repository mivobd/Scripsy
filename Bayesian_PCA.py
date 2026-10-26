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
from concurrent.futures import ProcessPoolExecutor

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



def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-2, file = ""):
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
        # printProgressBar(count, limit, suffix = f"Complete ({file})")
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

        print("alpha:", alphas_new)
        print("1 / alpha:", 1 / alphas_new)
        print("||w_i||^2:", np.sum(W_new*W_new, axis=0))

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

        A_diff = (
            np.linalg.norm(alphas_new - alphas)
            / max(np.linalg.norm(alphas), 1.0)
        )

        print(
            f"\ncount={count}, "
            f"W_diff={W_diff}, "
            f"beta_diff={beta_diff}, "
            f"A_diff={A_diff}"
        )

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
        printProgressBar(iteration, limit, suffix = f"Complete ({file})")
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

"""
Be forewarned, I made this code my own. This means my comments are not
strictly professional. They are still informative and should get the
point across just fine. But I also make jokes, because it makes things
fun and reading the code less tedious.
"""

"""
!!! CHECK BEFORE EACH SCRIPT EXECUTION !!!
"""
if __name__ == '__main__':
    snapshot = "N" # set the operating condition of the snapshot.   "N" for NOC and peruse fault_details.txt for the fault IDs
    control = 1 # boolean, control (1) or no control (0) for the snapshot
    autoreggressive_noise = 1
    measurement_noise = 1
    alpha = 0.05 # confidence level for the similarity checks
    failed_runs = ["run_89.json","run_75.json","run_56.json","run_413.json","run_403.json","run_385.json","run_376.json","run_344.json","run_330.json","run_326.json","run_303.json","run_269.json","run_266.json","run_231.json","run_183.json","run_160.json"] # runs where EM did not converge in iteration limit
    # pdr = [50, 51, 53, 54, 62, 71, 103] # Previously Detected Runs
    # pndr = [72, 66, 70, 69, 57] # Previously Not Detected Runs

    data = SimpleNamespace()
    data.dt = None # set the time difference manually (only needed if simulation time data not saved)
    snapshot_BPCA = False

    """
    Normal code execution below
    """
    folder_path = f"BPCA_H_C{control}_AR{autoreggressive_noise}_M{measurement_noise}"
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    operating_conditons = pd.read_csv('fault_details.txt', sep=',')["Fault ID"].to_numpy()
    operating_conditons = np.insert(operating_conditons, 0, "N")
    folder_name = f"H_C{control}_AR0_M{measurement_noise}"
    ## Snapshot data
    data.snapshot = SimpleNamespace() # dedicated to snapshot-related values
    stats = SimpleNamespace()
    with open(f"{folder_name}/OldSnapshot_{snapshot}.json", 'r', encoding='utf-8') as file:
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

    # Bayesian PCA
    limit = 1
    M = 13
    data.snapshot.mean = np.mean(data.set, axis=0)
    x = (data.set-data.snapshot.mean).T # scaling the data 

    if snapshot_BPCA == True:
        with ProcessPoolExecutor() as executor:
            global_search(x, M, limit, save_state = False, file = f"OldSnapshot_{snapshot}")
        shutil.copy2(f"save_state.json", f"{folder_path}/OldSnapshot_{snapshot}.json")

    # count = 0
    # while len(pdr) > 0:
    #     with open(f"{folder_name}/run_{pdr[-1]}.json", 'r', encoding='utf-8') as file:
    #         data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

    #     if count == 0:
    #         num.recorded_points = len(data.results["t"])
    #         num.measurements = len(data.results["t"]) - num.ignore        

    #     data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    #     # putting all the simulation data into one matrix
    #     for ii in range(0,num.variables):
    #         data.set[:, ii] = data.results[data.measured_variables[ii]]

    #     data.snapshot.mean, data.snapshot.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
    #     x = ((data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std).T # scaling the data

    #     global_search(x, M, limit, file = f"run_{pdr[-1]}.json")
    #     shutil.copy2(f"save_state.json", f"{folder_path}/run_{pdr[-1]}.json")
    #     pdr.pop()
    #     count += 1

    # while len(pndr) > 0:
    #     with open(f"{folder_name}/run_{pndr[-1]}.json", 'r', encoding='utf-8') as file:
    #         data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

    #     data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    #     # putting all the simulation data into one matrix
    #     for ii in range(0,num.variables):
    #         data.set[:, ii] = data.results[data.measured_variables[ii]]

    #     data.snapshot.mean, data.snapshot.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
    #     x = ((data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std).T # scaling the data

    #     global_search(x, M, limit, file = f"run_{pndr[-1]}.json")
    #     shutil.copy2(f"save_state.json", f"{folder_path}/run_{pndr[-1]}.json")
    #     pndr.pop()

    files = ["run_62.json"]
    # files = [f.name for f in Path(f"{folder_name}").iterdir() if f.is_file()]
    # files_complete = [f.name for f in Path(f"{folder_path}").iterdir() if f.is_file()] 
    # for i in range(len(files_complete)):
    #     files.remove(f"{files_complete[i]}")

    count = 0
    while len(files) > 1:
        with open(f"{folder_name}/{files[-1]}", 'r', encoding='utf-8') as file:
            data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

        if count == 0:
            num.recorded_points = len(data.results["t"])
            num.measurements = len(data.results["t"]) - num.ignore        

        data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
        # putting all the simulation data into one matrix
        for ii in range(0,num.variables):
            data.set[:, ii] = data.results[data.measured_variables[ii]]

        data.historical_mean = np.mean(data.set, axis=0)
        x = (data.set-data.historical_mean).T # scaling the data

        global_search(x, M, limit, file = f"{folder_name}/{files[-1]}")
        shutil.copy2(f"save_state.json", f"{folder_path}/{files[-1]}")
        files.pop()
        count += 1


    # with open(f"{folder_name}/run_463.json", 'r', encoding='utf-8') as file:
    #     data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)

    # if count == 0:
    #     num.recorded_points = len(data.results["t"])
    #     num.measurements = len(data.results["t"]) - num.ignore        

    # data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    # # putting all the simulation data into one matrix
    # for ii in range(0,num.variables):
    #     data.set[:, ii] = data.results[data.measured_variables[ii]]

    # data.snapshot.mean, data.snapshot.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
    # x = ((data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std).T # scaling the data

    # state, best_parameters = global_search(x, M, 1000, file = f"{folder_name}/run_54.json")
    # print(best_parameters[1])
    # shutil.copy2(f"save_state.json", f"{folder_path}/sigma2_test_463.json")



    # x = np.zeros([10, 300])
    # np.random.seed(1)

    # for d in range(0, 10):
    #     x[d, :] = np.random.normal(loc = 0, scale = 0.5, size = 300)
    # for d in [1, 3, 6]:
    #     x[d, :] = np.random.normal(loc = 0, scale = 1, size = 300)

    # x = x - np.mean(x, axis=1).reshape(-1, 1) # centering the data

    D = 10
    N = 300
    true_M = 3

    Z = np.random.normal(size=(true_M, N))
    W_true = np.random.normal(size=(D, true_M))
    noise = 0.25 * np.random.normal(size=(D, N))

    x = W_true @ Z + noise
    x -= np.mean(x, axis=1, keepdims=True)

    limit = 100
    M = 9

    with ProcessPoolExecutor() as executor:
        global_search(x, M, limit)
    shutil.copy2(f"save_state.json", f"test_results.json")

    results = SimpleNamespace()
    with open(f"test_results.json", 'r', encoding='utf-8') as file:
        results.Snapshot = SimpleNamespace(**json.load(file))


    # results = SimpleNamespace()
    # with open(f"{folder_path}/OldSnapshot_N.json", 'r', encoding='utf-8') as file:
    #     results.Snapshot = SimpleNamespace(**json.load(file))

    # with ProcessPoolExecutor() as executor:
    #     W_unordered, beta, alphas, count, seed = em(x, M)

    W_unordered = np.array(results.Snapshot.best_parameters[0])
    alphas = np.array(results.Snapshot.best_parameters[2])
    ordered_indices = np.argsort(1/alphas.reshape(-1))
    W = W_unordered[:, ordered_indices[::-1]]
    alphas_ordered = alphas[ordered_indices[::-1]]
    beta = results.Snapshot.best_parameters[1]

    # comparing original BPCA results to ordered and retained components
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    hinton(W_unordered, ax1)
    ax1.set_title("Unordered")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Directions")
    hinton(W, ax2)
    ax2.set_title("Ordered")
    ax2.set_xlabel("Features")
    # hinton(retained, ax3)
    # ax3.set_title("Retained")
    # ax3.set_xlabel("Features")
    plt.show()

    print(f"\n\nExecution time: {time.perf_counter() - timer_start} seconds")
    # plt.show()