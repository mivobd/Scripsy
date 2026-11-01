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
W = np.transpose(Vt[:k, :]) # shape(W) = (D, k), principal subspace, the number of rows of Vt corresponding to the chosen number of principal components
t = x@W # shape(t) = (N, D), score matrix, the projection of the input matrix onto the principal subspace
xbar_scaled = t@W.T # shape(t) = (N, D), projection of the the score matrix back onto the original variable space
e = x - xbar_scaled # shape(t) = (N, D), residuals or error, inter alia
stats.Q = np.sum(e**2, axis=1)
# print(stats.Q)
print(np.shape(e[0, :]))

# Q contributions not really meaningful but here they are anyway
# stats.Q_cont = np.sum([[r[row, variable] * r[row, variable] for variable in range(0, num.variables)] for row in range(0, num.measurements)], axis=0)

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


TEMP

# def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-4, file = ""):
#     """
#     Expectation-Maximization algorithm for Bayesian PCA

#     Parameters
#         x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
#         M : int, number of retained principal components
#         seed : int (optional), set to zero for reproducibility, otherwise set to any integer (preferably the counter in the global maximum search)
#         ads : int (optional), 0 for no automatic dimensionality selection (ADS) which returns normal the probabalistic PCA solution, 1 to enable ADS which returns the Bayesian PCA solution
#         limit : int (optional), the maximum allowable iterations
#         diff : float (optional), the maximum allowable difference between the current and previous iteration of the parameters W and beta

#     Returns
#         W_new : numpy array [DxM], the final estimate of the weight matrix
#         beta_new : float, the final estimate of the precision of the noise
#         hyper_new : numpy array [Mx1], the final estimate of the hyperparameters
#         count : int, the number of iterations performed
#     """
#     D, N = np.shape(x)
#     count = 0
#     hyper = 0
#     W = 0
#     beta = 0
#     E_z = np.zeros([M, N])
#     E_zzT = np.zeros([M, M, N])

#     np.random.seed(seed)
#     hyper_new = np.ones([M]).reshape(-1, 1)
#     W_new = np.random.rand(D,M)/np.random.rand(D,M)
#     beta_new = 1/(np.random.rand()/np.random.rand())

#     W_diff = np.sum((W_new-W)**2)
#     sigma2_diff = (beta_new-beta)**2
    
#     while count < limit and (W_diff > diff or sigma2_diff > diff ):
#         printProgressBar(count, limit, suffix = f"Complete ({file})")
#         # Updating parameters
#         W = W_new.copy()
#         beta = beta_new
#         W_sum1 = 0
#         W_sum2 = 0
#         I_M = np.identity(M)
#         M_matrix = W.T @ W + I_M/beta
#         M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)

#         # Expecation
#         for n in range(0, N):
#             E_z[:, n:n+1] = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT[:, :, n] = M_inv/beta + E_z[:, n:n+1] @ E_z[:, n:n+1].T

#             W_sum1 += x[:, n:n+1] @ E_z[:, n:n+1].T
#             W_sum2 += E_zzT[:, :, n]


#         # Maximisation
#         with np.errstate(divide='ignore', over='ignore'):
#             for i in range(0, M):
#                 hyper_new[i:i+1] = D /( W[:, i:i+1].T @ W[:, i:i+1] )
#                 # hyper_new[i:i+1] = D/(W[i:i+1, :]@W[i:i+1, :].T)

#         W_new = W_sum1 @ np.linalg.inv( W_sum2 + ads/beta*np.diag(hyper_new.reshape(M)) )

#         beta_new = 0
#         for n in range(0, N):
#             beta_new += N*D / ( x[:, n:n+1].T @ x[:, n:n+1] - 2 * E_z[:, n:n+1].T @ W_new.T @ x[:, n:n+1] + np.trace( E_zzT[:, :, n] @ W_new.T @ W_new) )

#         W_diff = np.sum((W_new-W)**2)
#         beta_diff = (beta_new-beta)**2
#         count += 1

#     return [W_new, beta_new, hyper_new, count, seed]

# def em(x, M, seed = 0, ads = True, limit = 1e4, diff = 1e-4, file = ""):
#     """
#     next to try using
#     linalg.solve(B.T, A.T).T
#     to find A@inv(B) instead of using cholesky
#     """
#     count = 0
#     D, N = np.shape(x)
#     # np.random.seed(seed)

#     W = 0
#     W_new = np.random.rand(D,M)/10
#     # W_new = np.ones([D,M])
#     W_diff = np.sum((W_new-W)**2)

#     beta = 0
#     beta_new = (np.random.rand()/np.random.rand())*100
#     # beta_new = 1
#     beta_diff = (beta_new-beta)**2

#     A = 0
#     A_new = np.diag(np.ones([M]))
#     A_diff = np.sum((A_new-A)**2)

#     while count < limit and ( max(W_diff, beta_diff, A_diff) > diff ):
#         printProgressBar(count, limit, suffix = f"Complete ({file})")
#         W = W_new.copy()
#         beta = beta_new
#         A = A_new.copy()

#         I_M = np.identity(M)
#         M_matrix = W.T @ W + I_M/beta
#         # M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)
#         M_inv = np.linalg.solve(M_matrix.T, I_M.T).T

#         W_sum1 = 0
#         W_sum2 = 0
#         for n in range(0, N):
#             E_z = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT = M_inv/beta + E_z @ E_z.T

#             W_sum1 += x[:, n:n+1] @ E_z.T
#             W_sum2 += E_zzT
#         # W_new = W_sum1 @ la.cho_solve(la.cho_factor(W_sum2 + A/beta, lower=True), I_M)
#         W_new = np.linalg.solve((W_sum2 + A/beta).T, W_sum1.T).T
#         # print(W_new - W_new1)

#         beta_sum1 = 0
#         beta_sum2 = 0
#         beta_sum3 = 0
#         for n in range(0, N):
#             E_z = M_inv @ W.T @ x[:, n:n+1]
#             E_zzT = M_inv/beta + E_z @ E_z.T

#             beta_sum1 += x[:, n] @ x[:, n]
#             beta_sum2 += E_z.T @ W_new.T @ x[:, n:n+1]
#             beta_sum3 += np.trace( E_zzT @ W_new.T @ W_new)
#         # print(beta_sum1)
#         beta_new = N*D / ( beta_sum1 - 2 * beta_sum2 + beta_sum3 )

#         for i in range(0, M):
#             A_new[i, i] = D / ( W_new[:, i] @ W_new[:, i] )

#         W_diff = np.sum((W_new-W)**2)
#         beta_diff = (beta_new-beta)**2
#         A_diff = np.sum((A_new-A)**2)
#         count += 1

#         # print(W_diff)
#         # print(beta_diff)
#         # print(A_diff)

#     return [W_new, beta_new, A_new, count, seed]

# def log_likelihood(x, parameters):
#     """
#     Computes the log-likelihood of the complete data given the parameters of the model.

#     Parameters
#         x : numpy array [DxN], consisting of the mean-centered observation column vectors (i.e. each column is one observation of each row)
#         parameters : list, containing the weight matrix W and the variance of the noise beta in the first and second elements, respectively

#     Returns
#         log_likelihood_value : float, the log-likelihood of the complete data given the parameters (single value but calculated as a numpy array, so sum is used so it is returned as a float)
#     """
#     W = parameters[0]
#     beta = parameters[1]
#     D, N = np.shape(x)
#     M = np.shape(W)[1]
#     x_mean = np.mean(x, axis=1).reshape(-1, 1)
#     I_M = np.identity(M)
#     M_matrix = W.T @ W + I_M/beta
#     M_inv = la.cho_solve(la.cho_factor(M_matrix, lower=True), I_M)

#     E_z = np.zeros([M, N])
#     E_zzT = np.zeros([M, M, N])
#     log_sum1 = 0
#     log_sum2 = 0
#     log_sum3 = 0
#     log_sum4 = 0

#     for n in range(0, N):
#         E_z[:, n:n+1] = (M_inv@W.T)@x[:, n:n+1]
#         E_zzT[:, :, n] = M_inv/beta + E_z[:, n:n+1]@E_z[:, n:n+1].T

#         log_sum1 += np.trace(E_zzT[:, :, n])
#         log_sum2 += x[:, n:n+1].T@x[:, n:n+1]
#         log_sum3 +=  E_z[:, n:n+1].T @ W.T @ x[:, n:n+1]
#         log_sum4 += np.trace( (E_zzT[:, :, n] @ W.T) @ W)

#     log_liklihood_value = - 1/2 * ( N*D * np.log( 2*np.pi/beta ) + log_sum1 + beta * ( log_sum2 - 2 * log_sum3 + log_sum4) )

#     return np.sum(log_liklihood_value)