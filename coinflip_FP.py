import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
# from scipy.integrate import trapz

# Function definitions
def factorial(num:int): # calculating num! = num*(num-1)*(num-2)*...*3*2*1
    return num*factorial(num-1) if num > 1 else num if num > 0 else 1

def combination(pool:int, sample:int): # combinations of size 'sample' from options 'pool'
    return factorial( pool )/factorial( sample )/factorial( pool - sample )

def beta(a:float, b:float, mean:float): # computes the probability of a mean from a beta distribution governed by hyperparameters a and b
    return gamma( a+b )/gamma( a )/gamma( b )*mean**( a - 1 )*( 1 - mean )**( b - 1)

def binomial(trials:int, successes:int, mean:float): # computes the probability of a number of successes using binomial distribution considering the number of trials and the mean
    return combination( trials, successes )*mean**successes*( 1-mean )**( trials-successes ) 

def inference(a:float, b:float, successes:int, trials:int): # computes the posterior distribution of a beta distribution after observing a number of successes and trials
    return beta( a+successes, trials-successes+b, np.linspace( 0, 1, 100 ) )

if __name__ == "__main__":
    colours = ['#e6194B', '#f58231', '#ffe119', '#bfef45', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6', '#a9a9a9'] # to show the learning
    
    mu = np.round(np.random.rand(1), decimals=2) # generating the coin's weighting (true theta)
    x = np.random.rand(100) # generating random numbers between 0 and 1 to serve as the coin tosses
    theta_range = np.linspace( start=0, stop=1, num=100 ) # range of theta values to plot the distributions over
    dtheta = theta_range[1] - theta_range[0]

    prior_HP = [1, 1] # setting the hyperparameters of the prior distribution to be uniform (Beta(1, 1))
    distribution = beta( prior_HP[0], prior_HP[1], theta_range ) # setting the prior distribution to be uniform (Beta(1, 1))
    theta_hat = np.round(prior_HP[0]/( prior_HP[0] + prior_HP[1] ), decimals=2) # calculating the mean of the prior distribution to serve as our initial guess of the coin's weighting

    # label = 'Prior, $\hat{\u03B8}$'+f'={theta_hat}'
    plt.plot( theta_range, distribution, color='#9A6324', label='Prior, $\hat{\u03B8}$'+f'={theta_hat}') # plotting our initial beliefs of the coin's weighting

    success_counter = 0

    for i in range(len(x)): # performing inference for each coin toss
        if x[i] < mu: # if the coin toss is a success
            success_counter += 1
        elif x[i] >= mu: # if the coin toss is a failure
            pass
        else:
            raise ValueError("The true theta could not be compared to the coin toss. Investigate values assigned to x and mu.")
        
        distribution = binomial( i+1, success_counter, theta_range )*distribution/np.trapezoid(binomial( i+1, success_counter, theta_range )*distribution, dx=dtheta) # calculating the posterior
        v = np.var(distribution); # variance of the posterior
        m = np.mean(distribution); # mean of the posterior 
        prior_HP[0] = m*v # recalculating a
        prior_HP[1] = ( 1 - m )*v # recalculating b
        theta_hat = np.round(prior_HP[0]/( prior_HP[0] + prior_HP[1] ), decimals=2)
        

        learning_tests = np.linspace( 0, len(x), 10, dtype=int ) # to show the learning after certain coin tosses
        if i in learning_tests: # plotting the posterior distribution after certain coin tosses to show the learning
            plt.plot(theta_range, distribution, color=colours[np.where(learning_tests == i)[0][0]], label=f'{i+1} tosses,'+' $\hat{\u03B8}$'+f'={theta_hat}') 

plt.title(f'True theta is {mu}\nThe number of heads landed is {sum( i < mu for i in x )[0]}')#\nThe final estimate for theta is {theta_hat:.2f}')
plt.plot( [mu, mu], [0, max( distribution )], linestyle=':', color='#000000', label='True theta') # presenting the true theta on the plot
plt.xlabel('p(heads)=\u03B8')
plt.ylabel('Probability density')
plt.legend()
plt.show()