import numpy as np
import matplotlib.pyplot as plt

# define a squared exponential covariance function
def squared_exponential(x1, x2, h):
    return np.exp(-0.5 * (x1 - x2) ** 2 / h ** 2)


np.random.seed(1)
x = np.linspace(0, 10, 100)
h = 1.0

mu = np.zeros(len(x))
C = squared_exponential(x, x[:, None], h)
draws = np.random.multivariate_normal(mu, C, 3)

print x.shape
print draws.shape

# first: plot a selection of unconstrained functions
# Plot the diagrams
plt.plot(x, draws.T, 'x')
plt.axis('equal')
plt.show()