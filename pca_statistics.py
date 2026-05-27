from types import SimpleNamespace
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
import os
os.system('cls')

alpha = 0.05
# shape(data) = (m, n), n variables observed m times
data = np.array([[7, 26, 6, 60], [1, 29, 15, 52], [11, 56, 8, 20], [11, 31, 8, 47], [7, 52, 6, 33], [11, 55, 9, 22], [3, 71, 17, 6], [1, 31, 22, 44], [2, 54, 18, 22], [21, 47, 4, 26], [1, 40, 23, 34], [11, 66, 9, 12], [10, 68, 8, 12]], dtype=np.float64)
x_mean = np.mean(data, axis=0) # shape(x_mean) = (1, n)
x_std = np.std(data, axis=0, ddof=1) # shape(x_std) = (1, n)
x = (data - x_mean) # shape(x) = (m, n)
U,S,Vt = np.linalg.svd(x, full_matrices=False)
# shape(U) = (m, n), unitary matrix of left singular vectors
# shape(S) = (1, n), singular values
# shape(Vt) = (n, n), unitary matrix of right singular vectors transposed
a = 2
num = SimpleNamespace()
num.measurements = len(x) # rows
num.variables = len(x[0]) # columns

stats = SimpleNamespace()

## Basic PCA stats
# # singular values
# print("singular values = \n\n", S, "\n")

# # coeff (loading matrix)
# print("loading matrix (P or W or coeff, inter alia) = \n\n", P, "\n")

# # score matrix
# print("score = \n\n", t, "\n")

# # latent (covariance matrix corrected)
# print("latent = \n\n", S**2 / (num.measurements-1), "\n")

# # eigenvalues
# print("eigenvalues = \n\n", S**2, "\n")

# # explained
# print("explained = \n\n", 100 * S**2 / (num.measurements-1) / np.sum(S**2 / (num.measurements-1)), "\n")


# Hotelling's T-squared statistic
# tsquared_MATLAB = np.sum(t**2/(S**2/(num.measurements-1)), axis=1) # equivalent to MATLAB result, but may not be correct with our references (which use the eigenvalues instead of the covariance matrix)
# print(tsquared_MATLAB)

P = np.transpose(Vt[:a, :]) # shape(P) = (n, a), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
t = x@P # I could put this in the line below but that would mean calculating this twice, it is faster to store it to memory and reuse it
covariance = S[:a]**2/(num.measurements-1) # covariance matrix
stats.tsquared = np.array([ t[row, :] @ np.diag( covariance**-1 ) @ t[row, :].T for row in range(0, num.measurements) ]) # strictly, t.T does nothing here because of how numpy defaults to handling 1D arrays
print(f"tsquared = \n\n{stats.tsquared}\n\n")

# Critical value
stats.tsquared95 = (a*(num.measurements-1)/(num.measurements-a)) * f.ppf(1-alpha, a, num.measurements - a) 
# tsquared95_normalised = stats.tsquared95**-a
# print(stats.tsquared95)

# stats.tsquared_contributions = np.sum([[np.sum([ 0 if (t[row, component] / S[component])**2*(num.measurements-1) < stats.tsquared95**-a else t[row, component] / ( S[component]**2/(num.measurements-1) ) * P[variable,component] * x[row,variable] for component in range(0, a)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)
# print(stats.tsquared_contributions)

# # not summed versions to verify:
# stats.tsquared_contributions = [[np.sum([ max((t[row, component] / S[component])**2 - stats.tsquared95**-a*0, 0) / t[row, component] * P[variable,component] * x[row,variable] for component in range(0, a)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)]
# # above is the same as below (more readable below)
stats.tsquared = stats.tsquared*0
stats.tsquared_contributions = np.zeros([num.variables])
for variable in range(0, num.variables):
    for row in range(0, num.measurements):
        for component in range(0, a):
            t_normalised = t[row, component]**2 / ( S[component]**2/(num.measurements-1) )
            if t_normalised < stats.tsquared95**-a:
                t_normalised = 0 # zero if not significant
            contribution = t_normalised / t[row, component] * P[variable,component] * x[row,variable]
            stats.tsquared[row] += contribution
            stats.tsquared_contributions[variable] += contribution
        
# stats.tsquared = np.sum(stats.tsquared_contributions, axis=1)
# print(f"tsquared by contribution = \n\n{stats.tsquared}\n\n")
# print(stats.tsquared_contributions)
# print(stats.tsquared)

# plt.bar(range(0, num.variables), stats.tsquared_contributions)
# plt.show()


## Q statistic
P = np.transpose(Vt[:a, :]) # shape(P) = (n, a), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
t = x@P # shape(t) = (m, n), score matrix, the projection of the input matrix onto the principal subspace
xbar_scaled = t@P.T # shape(t) = (m, n), projection of the the score matrix back onto the original variable space
e = x - xbar_scaled # shape(t) = (m, n), residuals or error, inter alia
stats.Q = np.sum(e**2, axis=1)
# print(f"Summing (x-x_bar)(x-x_bar) = \n\n{stats.Q}\n")

# below is the preferred method
P = np.transpose(Vt[:a, :]) # shape(P) = (n, a), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
r = np.array([ x[row, :] @ (np.identity(num.variables) - P@P.T) for row in range(0, num.measurements) ]) # identical to e
stats.Q = np.array([r[row, :].T@r[row, :] for row in range(0, num.measurements)]) # strictly, like before, r.T changes nothing in terms of execution
# print(f"Preferred method = \n\n{Q}\n\nManual version = \n\n{np.sum(e**2, axis=1)}\n\n")
# print(f"Preferred method = \n\n{stats.Q}\n")

# Q contributions not really meaningful but here they are anyway
# stats.Q_cont = np.sum([[r[row, variable] * r[row, variable] for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

# plt.bar(range(0, num.variables), stats.Q_cont)
# plt.show()

# Critical value
# residual_eigenvals = S[a:]**2
# theta = [np.sum(residual_eigenvals**ii) for ii in range(1, 4)]
# h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1]**2) # h0 value
# stats.Q95 = theta[0] * ( norm.ppf(1-alpha) * h0 * np.sqrt( 2*theta[1] )/theta[0] + 1 + theta[1]*h0*(h0-1)/theta[0]**2 ) / h0
# print(stats.Q95)

## Limit violations
# stats.r95 = binom.ppf(1-alpha, 1024, alpha) # critical value for T-squared and Q statistic violations, with 1024 observations