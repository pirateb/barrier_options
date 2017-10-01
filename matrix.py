import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.stats import norm


def gaussian_random_numbers(corr: np.ndarray, num_samples: int):
    # To create gaussian correlated random numbers need to take cholesky decomposition
    dimension = corr.shape[0]

    #X = norm.rvs(size=(dimension, num_samples))
    X = np.random.normal(size=(dimension, num_samples))
    L = cholesky(corr, lower=True)
    return np.dot(L, X)


def random_covariance(size: int):
    corr = np.tril(np.random.rand(size, size))
    np.fill_diagonal(corr, 1)
    corr = (corr + corr.T) / 2
    D = np.diag(np.sqrt(np.random.rand(size)*0.2))
    return np.dot(np.dot(D, corr), D)


def covariance_from_corr_std(rho: float, std: float):
    corr = np.array([[1.0, rho], [rho, 1.0]])
    return corr_2_cov(corr, std)


def cov_2_corr(cov):
    DInv = np.diag(1.0 / np.sqrt(np.diag(cov)))
    return np.dot(np.dot(DInv, cov), DInv)


def corr_2_cov(corr, sigma):
    D = np.eye(corr.shape[0]) * sigma
    return np.dot(np.dot(D, corr), D)