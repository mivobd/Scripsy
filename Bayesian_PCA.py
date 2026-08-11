## Housekeeping
from types import SimpleNamespace
import json
import os
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
os.system('cls')

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

def log_likelihood(x: np.ndarray, mu: np.ndarray, C: np.ndarray, S: np.ndarray, W: np.ndarray, sigma2: np.ndarray, N: int, D: int):
    M = W.T@W + sigma2*np.eye(D)
    C_inv = np.sqrt(sigma2)^-1*np.eye(D) - sigma2**-1 @ W @ np.linalg.inv(M) @ W.T
    return - N/2 * ( D*np.log(2*np.pi) + np.log(np.linalg.det(C)) + np.trace(C_inv @ S) )

# def prob_x(mu: np.ndarray, S: np.ndarray, D: int):
#     return 1/(2*np.pi)**D/2 * 1/np.linalg.det(S)**0.5 * np.exp(-0.5 * (x - mu).T @ np.linalg.inv(S) @ (x - mu))

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

# PCA
data.snapshot.mean, data.snapshot.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
x = (data.set[num.ignore:, :]-data.snapshot.mean)/data.snapshot.std # scaling the data


prob = SimpleNamespace()
# np.random.multivariate_normal(mean, covar)

I = np.identity(num.variables)
# M = W.T@W + sigma^2*I

# prob.z = np.random.multivariate_normal(0, I)
# prob.x_given_z = np.random.multivariate_normal(W@z+mu, sigma^2@I)
# prob.x = np.random.multivariate_normal(data.snapshot.mean, data.set[num.ignore:, :]-data.snapshot.mean)
# prob.z_given_x = np.random.multivariate_normal(np.linalg.inv(M)@W.T@(x-mu),sigma^-2*M)

# EM algorithm

count = 0
W = 0
sigma2 = 0
W_new = np.ones(num.variables)
sigma2_new = 1

x_mean = data.snapshot.mean

while count < 100000 or ( max((W_new-W)^2) > 1e-3 and max((sigma2_new-sigma2)^2) > 1e-3 ):
    # Updating parameters
    W = W_new
    sigma2 = sigma2_new
    M = W.T@W + sigma2*I

    # Expecation 
    E_z = np.array([np.linalg.inv(M)@W.T@(x[measurement, :]-x_mean) for measurement in range(0, num.measurements)])
    E_zzT = np.array([sigma2*np.linalg.inv(M) + E_z[measurement]*E_z[measurement].T for measurement in range(0, num.measurements)])

    # Maximisation
    W_new = np.sum([(x[measurement, :] - x_mean)*E_z[measurement].T for measurement in range(0, num.measurements)])*np.sum(E_zzT)**-1
    sigma2_new = 1/(num.measurements*num.variables) * np.sum([np.linalg.norm(x[measurement, :]-x_mean)**2 - 2*E_z[measurement].T*W_new.T*(x[measurement, :]-x_mean) + np.trace(E_zzT[measurement]*W_new.T@W_new) for measurement in range(0, num.measurements)])


# C = W@W.T + sigma2*np.eye(D)

