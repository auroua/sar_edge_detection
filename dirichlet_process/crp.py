# Generate table assignments for `num_customers` customers, according to
# a Chinese Restaurant Process with dispersion parameter `alpha`.
# returns an array of integer table assignments

# distribution sampled from a Dirichlet process
import numpy as np
from scipy.stats import dirichlet, beta, norm
import matplotlib.pyplot as plt
from numpy.random import choice
import pandas as pd


def dirichlet_sample_approximation(base_measure, alpha, tol=0.01):
    betas = []
    pis = []
    betas.append(beta(1, alpha).rvs())
    pis.append(betas[0])
    while sum(pis) < (1.-tol):
        s = np.sum([np.log(1 - b) for b in betas])
        new_beta = beta(1, alpha).rvs()
        betas.append(new_beta*np.exp(s))
        pis.append(new_beta*np.exp(s))
    pis = np.array(pis)
    thetas = np.array([base_measure() for _ in pis])
    return pis, thetas


def plot_normal_dp_approximation(alpha):
    plt.figure()
    plt.title("Dirichlet Process Sample with N(0,1) Base Measure")
    plt.suptitle("alpha: %s" % alpha)
    pis, thetas = dirichlet_sample_approximation(lambda: norm().rvs(), alpha)
    pis = pis * (norm.pdf(0) / pis.max())
    print pis
    plt.vlines(thetas, 0, pis, )
    X = np.linspace(-4,4,100)
    plt.plot(X, norm.pdf(X))
    plt.show()

def chinese_restaurant_process(num_customers, alpha):
    if num_customers <= 0:
        return []

    table_assignment = [1]
    next_open_table = 2
    for i in range(1, num_customers):
        if np.random.rand(1) < alpha*1.0/(i+alpha):
            table_assignment.append(next_open_table)
            next_open_table += 1
        else:
            table = table_assignment[np.random.randint(len(table_assignment))]
            table_assignment.append(table)
    print table_assignment
    return table_assignment


def polya_urn_model(num_balls, alpha):
    if num_balls <= 0:
        return []
    balls_in_urn = [round(np.random.randn(1), 2)]
    for i in range(1, num_balls):
        if np.random.rand(1) < alpha*1.0/(i + alpha):
            new_color = round(np.random.randn(1), 2)
            balls_in_urn.append(float(new_color))
        else:
            ball = balls_in_urn[np.random.randint(len(balls_in_urn))]
            balls_in_urn.append(ball)
    print balls_in_urn
    return balls_in_urn


class DirichletProcessSample(object):
    def __init__(self, base_measure, alpha):
        self.base_measure = base_measure
        self.alpha = alpha

        self.cache = []
        self.weights = []
        self.total_stick_used = 0

    def __call__(self, *args, **kwargs):
        remaining = 1.0 - self.total_stick_used
        i = DirichletProcessSample.roll_dice(self.weights + [remaining])
        if i is not None and i < len(self.weights):
            return self.cache[i]
        else:
            stick_piece = beta(1, self.alpha).rvs() * remaining
            self.total_stick_used += stick_piece
            self.weights.append(stick_piece)
            new_value = self.base_measure()
            self.cache.append(new_value)
            return new_value

    @staticmethod
    def roll_dice(weights):
        if weights:
            return choice(range(len(weights)), p=weights)
        else:
            return None

if __name__ == '__main__':
    # chinese_restaurant_process(10, 5)
    # polya_urn_model(10, 0.3)
    # plot_normal_dp_approximation(100)

    base_measure = lambda: norm().rvs()
    n_samples = 10000
    samples = {}
    for alpha in [1, 10, 100, 1000]:
        dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
        samples["Alpha: %s" % alpha] = [dirichlet_norm() for _ in range(n_samples)]

    _ = pd.DataFrame(samples).hist()
    plt.show()