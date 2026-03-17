import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4,4,1000)
mu = 0
sigma = 1
p = ( 2*np.pi ) ** ( -0.5 ) / sigma*np.exp( -0.5*( ( x-mu )/sigma )**2 )

plt.plot(x, p, '-')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Normal Distribution')
plt.show()