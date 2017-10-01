import numpy as np
import matplotlib.pyplot as plt

from matrix import gaussian_random_numbers, random_covariance, cov_2_corr


def simulate_stock_movements_brownian(cov, T, steps, r):
    sigma = np.diag(cov)**0.5
    sigma = sigma.reshape((len(sigma), 1))

    drift = r - 0.5 * (sigma**2)
    dt = T / steps
    sqrt_dt = dt ** 0.5

    corr = cov_2_corr(cov)
    random_seq = gaussian_random_numbers(corr=corr, num_samples=steps)

    return drift*dt + sqrt_dt* np.multiply(sigma, random_seq)


def plot_sample_stock_movements(cov):
    T = 1
    steps = 200
    r = 0.05

    stock_movements = simulate_stock_movements_brownian(cov, T, steps, r)
    stock_movements = stock_movements.cumsum(axis=1).T
    plt.plot(stock_movements)
    plt.show()
