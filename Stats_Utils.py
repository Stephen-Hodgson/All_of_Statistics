import numpy as np
from scipy.stats import norm

def non_para_bootstrap(X, f, B = 1000):
    n = len(X)
    Xboot = np.random.choice(X, size=(n, B))
    thetaboot = f(Xboot)
    se = np.std(thetaboot)
    return se

def norm_interval(thetahat, se, alpha = 0.05):
    return (thetahat + norm.ppf(alpha / 2) * se, thetahat + norm.ppf(1 - alpha / 2) * se)

def pivot_interval(thetahat, thetaboot, alpha = 0.05):
    return (2 * thetahat - np.quantile(thetaboot, 1 - alpha / 2), 2 * thetahat - np.quantile(thetaboot, alpha / 2))

def percentile_interval(thetaboot, alpha = 0.05):
    return (np.quantile(thetaboot, alpha / 2), np.quantile(thetaboot, 1 - alpha / 2))

def two_sample_difference(mean1, se1, mean2, se2, alpha=0.05):
    diff = mean1 - mean2
    se = np.sqrt(se1 ** 2 + se2 ** 2)
    z = - norm.ppf(alpha / 2)
    C = (diff - z * se, diff + z * se)
    return C

def prop_C(n, p, alpha = 0.05):
    se = np.sqrt(p * (1 - p) / n)
    z = - norm.ppf(alpha / 2)
    C = (p - z * se, p + z * se)
    return C