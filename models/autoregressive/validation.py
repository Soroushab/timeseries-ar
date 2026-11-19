import numpy as np
from models.autoregressive.simulation import simulate_ar
from models.autoregressive.theory import theoretical_autocovariances

def simulate_terminal_values(phi, sigma2=1.0,
                             n_paths=10000, n_steps=150,
                             seed=42):

    rng = np.random.default_rng(seed)

    initial_values = [0.0] * len(phi)

    last_values = []

    for i in range(n_paths):
        shocks = rng.normal(
            loc=0, scale=np.sqrt(sigma2), size=n_steps
        ).tolist()

        path = simulate_ar(phi, initial_values, shocks)

        last_values.append(path[-1])

    return np.array(last_values)



def calculate_moments(values):

    # Calculate mean and variance
    mean = np.mean(values)
    variance = np.var(values, ddof=1)

    # Standardize observations
    z = (values - mean) / np.std(values)

    # Calculate higher-order moments
    skewness = np.mean(z**3)
    kurtosis = np.mean(z**4)

    return np.array([mean, variance, skewness, kurtosis])




def validate_ar(phi, sigma2=1.0,
                n_paths=10000, n_steps=150,
                seed=42):

    # Calculate theoretical moments
    gamma = theoretical_autocovariances(phi, sigma2)

    theoretical = np.array([0.0, gamma[0], 0.0, 3.0])

    # Simulate independent paths
    values = simulate_terminal_values(
        phi, sigma2, n_paths, n_steps, seed
    )

    # Calculate empirical moments
    empirical = calculate_moments(values)

    # Calculate standard errors
    n = len(values)

    standard_errors = np.array([
        np.sqrt(gamma[0] / n),
        gamma[0] * np.sqrt(2 / (n - 1)),
        np.sqrt(6 / n),
        np.sqrt(24 / n)
    ])

    # Compare empirical and theoretical moments
    errors = np.abs(empirical - theoretical)

    tolerances = 3 * standard_errors

    passed = errors <= tolerances

    return passed
