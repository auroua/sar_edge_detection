import numpy as np
from random import uniform, gauss, choice, randint

# Global variables
noise_sigma = 100
sin_lam_max = 1
sin_bias_max = 1000

# Kernel Mathematical Sequences
def poisson_seq(n=10, lam=100, noise_k=0):
    seq = np.random.poisson(lam, n)
    return np.array(map(lambda x: int(x + noise_k * (np.random.normal(lam, noise_sigma) - lam)), seq))

def sin_seq(n=10, lam=2, phase=5, bias=0, K=100, noise_k=0):
    _x = np.linspace((0 + phase) * lam, (n + phase) * lam, n)
    seq = np.sin(_x) + 1
    return np.array(map(lambda x: int(K * x + bias + noise_k * (np.random.normal(K, noise_sigma) - K)), seq))

def linear_seq(n=10, K=2, b=10, noise_k=0):
    return np.array(map(lambda x: int(K * x + b + noise_k * (np.random.normal(b, noise_sigma) - b)), np.arange(n)))

# Complicated Compound Sequencs
# Every content sequence consist of one poisson component and several sin components
def continuous_signal_seq(n=10, lam=100, noise_k=1):
    print "Content seq:"
    print "----------------------------------------------------"
    k_list = list(decomposition(lam))
    print "The proportion of each component includes: ", k_list
    pop_index = randint(0, len(k_list) - 1)
    print "The proportion of poisson is: ", k_list[pop_index]
    seq = poisson_seq(n=n, lam=k_list.pop(pop_index), noise_k=noise_k)
    print "Poisson component is:\n ", seq

    for k in k_list:
        phase = uniform(-np.pi, np.pi)
        sin_lam = uniform(0, sin_lam_max)
        sin_bias = uniform(0, sin_bias_max)
        new_seq = sin_seq(n=n, lam=sin_lam, phase=phase, bias=sin_bias, K=k, noise_k=noise_k)
        seq += new_seq
        print "Sin component is (proportion:[%d]):\n " % k, new_seq

    return seq

# Datasets with one exception
# including several continuous signal sequences, and there is a exception at time 't' in each sequence.
def one_exception_dataset(N=10, n=10, T=[], lam=1000, exc=-1, noise_k=0):
    dataset   = np.array(map(lambda x: continuous_signal_seq(n=n, lam=lam, noise_k=noise_k), np.arange(N))).T
    for t in T:
        exception  = linear_seq(n=N, K=0, b=exc, noise_k=noise_k)
        dataset[t] = exception
    # print dataset
    return dataset


# Utils
def decomposition(i):
    while i > 0:
        n = randint(1, i)
        yield n
        i -= n

if __name__ == "__main__":
    seqs = []
    n = 100
    seqs.append(continuous_signal_seq(n=n, lam=8000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=7000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=6000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=1000, noise_k=1))
    # seqs.append(content_seq(n=n, lam=3000, noise_k=1))
    # for i in range(5):
    #     seqs.append(content_seq(n=n, lam=2000, noise_k=1))
    # plot_dataset(seqs, n=n)