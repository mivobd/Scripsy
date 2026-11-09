from types import SimpleNamespace
import numpy as np
from scipy.stats import f, norm, binom
import matplotlib.pyplot as plt
import os
os.system('cls')

alpha = 0.05
# shape(data) = (N, D), D variables observed N times
data = np.array([[7, 26, 6, 60], [1, 29, 15, 52], [11, 56, 8, 20], [11, 31, 8, 47], [7, 52, 6, 33], [11, 55, 9, 22], [3, 71, 17, 6], [1, 31, 22, 44], [2, 54, 18, 22], [21, 47, 4, 26], [1, 40, 23, 34], [11, 66, 9, 12], [10, 68, 8, 12]], dtype=np.float64)
x_mean = np.mean(data, axis=0) # shape(x_mean) = (1, D)
x_std = np.std(data, axis=0, ddof=1) # shape(x_std) = (1, D)
x = (data - x_mean) # shape(x) = (N, D)
U,S,Vt = np.linalg.svd(x, full_matrices=False)
# shape(U) = (N, D), unitary matrix of left singular vectors
# shape(S) = (1, D), singular values
# shape(Vt) = (D, D), unitary matrix of right singular vectors transposed
# print(np.shape(Vt))
k = 3
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

W = np.transpose(Vt[:k, :]) # shape(W) = (D, k), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
t = x@W # I could put this in the line below but that would mean calculating this twice, it is faster to store it to memory and reuse it
# print(np.shape(W))

covariance = S[:k]**2/(num.measurements-1) # covariance matrix
# stats.tsquared = np.array([ t[row, :] @ np.diag( covariance**-1 ) @ t[row, :].T for row in range(0, num.measurements) ]) # strictly, t.T does nothing here because of how numpy defaults to handling 1D arrays
stats.tsquared = np.zeros(num.measurements)
for row in range(0, num.measurements):
    stats.tsquared[row] = t[row, :] @ np.diag( covariance**-1 ) @ t[row, :].T
# print(stats.tsquared)
# print(np.diag(t @ np.diag( covariance**-1 ) @ t.T))

# print(f"tsquared = \n\n{stats.tsquared}\n\n")

# Critical value
stats.tsquared95 = (k*(num.measurements-1)/(num.measurements-k)) * f.ppf(1-alpha, k, num.measurements - k) 
# tsquared95_normalised = stats.tsquared95**-k
# print(stats.tsquared95)

# stats.tsquared_contributions = np.sum([[np.sum([ 0 if (t[row, component] / S[component])**2*(num.measurements-1) < stats.tsquared95**-k else t[row, component] / ( S[component]**2/(num.measurements-1) ) * W[variable,component] * x[row,variable] for component in range(0, k)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)
# print(stats.tsquared_contributions)

# # not summed versions to verify:
# stats.tsquared_contributions = [[np.sum([ max((t[row, component] / S[component])**2 - stats.tsquared95**-k*0, 0) / t[row, component] * W[variable,component] * x[row,variable] for component in range(0, k)], axis = 0) for variable in range(0, num.variables)] for row in range(0, num.measurements)]
# # above is the same as below (more readable below)
stats.tsquared = np.zeros(num.measurements)
stats.tsquared_contributions = np.zeros([num.variables])

for kk in range(0,1):
    # print(np.shape(t[0, :]))
    # print(np.shape(np.linalg.inv(np.diag(covariance))))
    # print(np.shape(W[0,:]))

    for variable in range(0, num.variables):
        for measurement in range(0, num.measurements):
            # for component in range(0, k):
            #     t_normalised = t[measurement, component]**2 / ( S[component]**2/(num.measurements-1) )
            #     if t_normalised < stats.tsquared95**-k:
            #         t_normalised = 0 # zero if not significant
            #     contribution = t_normalised / t[measurement, component] * W[variable,component] * x[measurement,variable]
            contribution = t[measurement, :] @ np.linalg.inv(np.diag(covariance)) @ W[variable,:] * x[measurement,variable]
            stats.tsquared[measurement] += contribution
            stats.tsquared_contributions[variable] += contribution
        
# stats.tsquared = np.sum(stats.tsquared_contributions, axis=1)
# print(f"tsquared by contribution = \n\n{stats.tsquared}\n\n")
# print(f"tsquared variable contributions = \n\n{stats.tsquared_contributions}\n\n")

# plt.bar(range(0, num.variables), stats.tsquared_contributions)
# plt.show()

## Q statistic
x = x.T
W = np.transpose(Vt[:k, :]) # shape(W) = (D, k), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
t = W.T @ x # shape(t) = (N, D), score matrix, the projection of the input matrix onto the principal subspace
xbar = W @ t # shape(t) = (N, D), projection of the the score matrix back onto the original variable space
e = x - xbar # shape(t) = (N, D), residuals or error, inter alia
stats.Q = np.sum(e**2, axis=0)
print(stats.Q)

# Q contributions not really meaningful but here they are anyway
stats.Q = np.zeros(num.measurements)
stats.Q_cont = np.zeros(num.variables)
for row in range(0, num.measurements):
    for variable in range(0, num.variables):
        contribution = e[variable, row] * e[variable, row]
        stats.Q_cont[variable] += contribution
        stats.Q[row] += contribution
# print(f"Q contributions = \n\n{stats.Q_cont}\n\n")
print(stats.Q)


# plt.bar(range(0, num.variables), stats.Q_cont)
# plt.show()

# Critical value
# residual_eigenvals = S[k:]**2
# theta = [np.sum(residual_eigenvals**ii) for ii in range(1, 4)]
# h0 = 1 - (2 * theta[0] * theta[2]) / (3 * theta[1]**2) # h0 value
# stats.Q95 = theta[0] * ( norm.ppf(1-alpha) * h0 * np.sqrt( 2*theta[1] )/theta[0] + 1 + theta[1]*h0*(h0-1)/theta[0]**2 ) / h0
# print(stats.Q95)

## Limit violations
# stats.r95 = binom.ppf(1-alpha, 1024, alpha) # critical value for T-squared and Q statistic violations, with 1024 observations

C = x.T@x/(num.measurements-1) # covariance matrix
eig_vals, eig_vecs = np.linalg.eig(C)
# print(np.diag(eig_vals))
# print(np.diag(S)@np.diag(S)/(num.measurements-1))
