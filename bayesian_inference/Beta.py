import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

a = 0.1
b = 0.1
mu = np.linspace( 0, 1, 100 )
p = gamma( a+b )/gamma( a )/gamma( b )*mu**( a - 1 )*( 1 - mu )**( b - 1)

plt.plot(mu, p, '-')
plt.xlabel('\u03BC')
plt.ylabel('p(\u03BC)')
plt.title(f'Beta Distribution, a = {a} and b = {b}')
plt.show()