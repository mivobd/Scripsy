## Housekeeping
from types import SimpleNamespace
import json
import os
import numpy as np
from scipy.stats import f, norm, binom
os.system('cls')

# Print iterations progress
def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
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
alpha = 0.05 # confidence level for the similarity checks

data = SimpleNamespace()
data.dt = None; # set the time difference manually (only needed if simulation time data not saved)

"""
Normal code execution below
"""
## Snapshot data
data.S = SimpleNamespace()
with open(f"S{control}/{snapshot}.json", 'r', encoding='utf-8') as file:
    data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
data.S.fault = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
data.measured_vars = data.results["measurements"]

num = SimpleNamespace()
num.recorded_points = len(data.results[data.measured_vars[0]])
# Getting the measurement time interval to slice out the window of the
# chosen operating condition (ignoring the NOC portion, if there is a
# disturbance)
data.t = data.results["t"]
data.dt = data.t[1] - data.t[0]

data.measured_vars.remove("t") # the measurement times are not being considered for PCA, removing it here saves effort later
num.variables = len(data.measured_vars); # the number of variables that will be in the matrix we perform PCA on

num.ignore = int(data.results["fault time"] / data.dt) # the number of measurements that will be ignored (corresponding to the period of NOC before the disturbance)
num.measurements = len(data.t) - num.ignore # the number of measurements that will be used for PCA (corresponding to the period after the disturbance)
data.set = np.zeros([num.recorded_points, num.variables]) # initialising the matrix that will have all the data to perform PCA

# putting all the simulation data into one matrix
for ii in range(0, num.variables):
    data.set[:, ii] = data.results[data.measured_vars[ii]]

# PCA
data.S.mean, data.S.std = np.mean(data.set[num.ignore:, :], axis=0), np.std(data.set[num.ignore:, :], axis=0, ddof=1) # calculating the mean and standard deviation of the data (ignoring the NOC portion)
x = (data.set[num.ignore:, :]-data.S.mean)/data.S.std # scaling the data
U, S, Vt = np.linalg.svd(x, full_matrices=False) # computing the SVD of the scaled data

V, explained = np.transpose(Vt), 100 * S**2 / (len(x)-1) / np.sum(S**2 / (len(x)-1)) # converting the SVD results into the PCA stats

# choosing the number of principle components based on how many are needed
# to explain 95 % of the variance
for ii in range(0, len(explained)):
    if np.sum(explained[:ii]) > 95:
        a = ii
        break

# PCA stats for the chosen principal components
L = V[:, :a] # principal subspace of the snapshot
residual_eigenvals = S[a:] # eigenvalues
theta = [np.sum(residual_eigenvals**ii) for ii in range(1, 4)] # theta values
h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1]**2) # h0 value

# Similarity checks
stats = SimpleNamespace()
stats.tsquared95 = ( a * (len(x) - 1) / (len(x) - a) ) * f.ppf(1 - alpha, a, len(x) - a) # 95 % confidence limit for Hotelling's T-squared
stats.Q95 = theta[0] * ( norm.ppf(1-alpha) * h0 * np.sqrt( 2*theta[1] )/theta[0] + 1 + theta[1]*h0*(h0-1)/theta[0]**2 ) / h0 # 95 % confidence limit for the Q statistic
stats.r95 = binom.ppf(1-alpha, len(x), alpha)
stats.S_pcaBest = 0.965 # Johannesmeyer provided best value

## Historical data
num.files = 1
while os.path.exists(f"H{control}/run_{num.files}.json"):
    num.files = num.files + 1
num.files = num.files - 1 # the number of data files

# Initialising arrays 
stats.tsquared_violations = np.zeros(num.files)
stats.tsquared_cont = np.zeros([num.files, num.variables])
stats.Q_violations = np.zeros(num.files)
stats.Q_cont = np.zeros([num.files, num.variables])
stats.S_pca = np.zeros(num.files)

data.H = SimpleNamespace()
data.H.faults = np.empty(num.files, dtype=np.dtype('U5'))
printProgressBar(0, num.files)
for kk in range(0,num.files):
    with open(f"H{control}/run_{kk+1}.json", 'r', encoding='utf-8') as file:
        data.results = json.load(file) # results = (keys: string), (values: 1x1 cell array: 1024x1 cell)
    data.H.faults[kk] = ''.join(chr(ii) for ii in (int(jj) for jj in data.results["fault"]))
    
    data.set = np.zeros((num.recorded_points, num.variables)) # initialising the matrix that will have all the data to perform PCA
    # putting all the simulation data into one matrix
    for ii in range(0,num.variables):
        data.set[:, ii] = data.results[data.measured_vars[ii]]

    # PCA                                                                                                               
    x = (data.set[num.ignore:, :]-data.S.mean)/data.S.std # scaling the data
    U, S, Vt = np.linalg.svd(x, full_matrices=False) # computing the SVD of the scaled data
    P = np.transpose(Vt[:a, :]) # shape(P) = (n, a), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
    
    # tsquared
    t = t = x@P # score matrix
    # use of eigenvalues deprecated for the time being (since covar is not being used for the tsquared)
    # eigenvals = S[:a]**2 # eigenvalues
    # covar = eigenvals/(len(x)-1) # covariance matrix
    stats.tsquared_violations[kk] = np.sum([int(ii) for ii in [ t[row, :] @ np.diag( S[:a]**-2 ) @ t[row, :].T for row in range(0, num.measurements) ] > stats.tsquared95]) # Hotelling's T-squared statistic violations
    
    stats.tsquared_cont[kk, :] = np.sum([[np.sum([ 0 if (t[row, component] / S[component])**2 < stats.tsquared95**-a else t[row, component] / S[component]**2 * P[variable,component] * x[row,variable] for component in range(0, a)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

    # Q statistic
    r = np.array([ x[row, :] @ (np.identity(num.variables) - P@P.T) for row in range(0, num.measurements) ]) # identical to e
    stats.Q_violations[kk] = np.sum([int(ii) for ii in np.array([r[row, :].T@r[row, :] for row in range(0, num.measurements)]) > stats.Q95]) # Q statistic violations
    # Q contributions not really meaningful
    # stats.Q_cont[kk, :] = np.sum([[r[row, variable] * r[row, variable] for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

    # PCA similarity statistic
    M = P
    stats.S_pca[kk] = np.trace(L.T@M@M.T@L) / a  

    printProgressBar(kk+1, num.files)

## Similarity stats
# A priori labelling for evaluating performance, not available for use
matches = data.H.faults == data.S.fault
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

print(f"Snapshot fault: {data.S.fault}\n")
for ii, statistic in enumerate([["tsquared_violations", stats.r95], ["Q_violations", stats.r95], ["S_pca", stats.S_pcaBest]]):
    candidate_pool = [int(ii) for ii in getattr(stats, statistic[0]) > statistic[1]]
    Np[ii] = np.sum(getattr(stats, statistic[0]) > statistic[1]) # candidate pool size (number of data sets classified as similar)
    N1[ii] = np.sum((getattr(stats, statistic[0]) > statistic[1]) * (matches)) # candidate pool data sets that are actually similar
    N2[ii] = np.sum((getattr(stats, statistic[0]) > statistic[1]) * (not_matches)) # candidate pool data sets that are not similar (N2 = Np - N1)
    p[ii] = N1[ii] / Np[ii] * 100
    n[ii]= N1[ii] / Ndb * 100
    nmax[ii] = np.min([Np[ii]/Ndb, 1]) * 100
    e[ii] = ( p[ii] + n[ii] ) / 2

    print(f"\n{statistic[0]} Results:")
    print(f"Np     {Np[ii]}\nN1     {N1[ii]}\nN2     {N2[ii]}\nNdb    {Ndb}\np      {p[ii]}\nn      {n[ii]}\nnmax   {nmax[ii]}\ne      {e[ii]}")

