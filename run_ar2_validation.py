
import numpy as np
from models.autoregressive.simulation import simulate_ar

# Initialize the random number generator
rng = np.random.default_rng(43)

# Define AR(2) parameters
phi = [0.4, 0.2]
initial_values = [0.0, 0.0]

# Monte Carlo simulation
last_values = []

for i in range(10000):
    shocks = rng.normal(loc=0, scale=1, size=150).tolist()

    path = simulate_ar(phi, initial_values, shocks)

    last_values.append(path[-1])

# Calculate empirical moments
empirical_mean = np.mean(last_values)
empirical_variance = np.var(last_values, ddof=1)

# Display empirical results
print("Empirical mean:", empirical_mean)
print("Empirical variance:", empirical_variance)


# Extract AR(2) coefficients
phi1 = phi[0]
phi2 = phi[1]

# Construct the system of equations
A = np.array([
    [1 - phi1**2 - phi2**2, -2 * phi1 * phi2],
    [-phi1, 1 - phi2]
])

b = np.array([1.0, 0.0])

# Solve the system
gamma0, gamma1 = np.linalg.solve(A, b)

# Define theoretical moments
theoretical_mean = 0.0
theoretical_variance = gamma0

# Display theoretical results
print("Theoretical mean:", theoretical_mean)
print("Theoretical variance:", theoretical_variance)
print("Lag-1 autocovariance:", gamma1)





# Calculate higher-order moments
values = np.array(last_values)

z = (values - np.mean(values)) / np.std(values)

empirical_skewness = np.mean(z**3)
empirical_kurtosis = np.mean(z**4)

# Define theoretical higher-order moments
theoretical_skewness = 0.0
theoretical_kurtosis = 3.0

# Calculate absolute errors
mean_error = abs(empirical_mean - theoretical_mean)
variance_error = abs(empirical_variance - theoretical_variance)
skewness_error = abs(empirical_skewness - theoretical_skewness)
kurtosis_error = abs(empirical_kurtosis - theoretical_kurtosis)

# Calculate standard errors
n = len(last_values)

mean_se = np.sqrt(theoretical_variance / n)
variance_se = theoretical_variance * np.sqrt(2 / (n - 1))
skewness_se = np.sqrt(6 / n)
kurtosis_se = np.sqrt(24 / n)

# Define statistical tolerances
mean_tolerance = 3 * mean_se
variance_tolerance = 3 * variance_se
skewness_tolerance = 3 * skewness_se
kurtosis_tolerance = 3 * kurtosis_se

# Validate statistical moments
mean_passed = mean_error <= mean_tolerance
variance_passed = variance_error <= variance_tolerance
skewness_passed = skewness_error <= skewness_tolerance
kurtosis_passed = kurtosis_error <= kurtosis_tolerance

# Display validation results
print("Empirical skewness:", empirical_skewness)
print("Empirical kurtosis:", empirical_kurtosis)

print("Mean validation passed:", mean_passed)
print("Variance validation passed:", variance_passed)
print("Skewness validation passed:", skewness_passed)
print("Kurtosis validation passed:", kurtosis_passed)
