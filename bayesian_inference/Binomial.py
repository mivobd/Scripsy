import numpy as np
import matplotlib.pyplot as plt

def factorial(num:int): # calculating num! = num*(num-1)*(num-2)*...*3*2*1
    return num*factorial(num-1) if num > 1 else num if num > 0 else 1

def combination(pool:int, sample:int): # combinations of size 'sample' from options 'pool'
    return factorial( pool )/factorial( sample )/factorial( pool - sample )

N = 10
mu = 0.5 # flipping a coin
m = np.linspace(0, N, N+1) # getting 0-N sides
p=np.ones(len(m))
for i in range(len(m)):
    p[i] = combination( N, m[i] )*mu**m[i]*( 1-mu )**( N-m[i] )

plt.plot(m, p, '-')
plt.xlabel('m')
plt.ylabel('p(m)')
plt.title('Binomial Distribution')
plt.show()