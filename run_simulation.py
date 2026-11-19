
# Import dependencies
import numpy as np
from models.autoregressive.simulation import simulate_ar


# Initialize the random number generator
rng = np.random.default_rng(42)


# Experiment 1: Single-path simulation
shocks = rng.normal(loc=0, scale=1, size=10).tolist()

x = simulate_ar(
    phi=[0.6],
    initial_values=[0.0],
    shocks=shocks
)

# Display the simulated path
print("Shocks:", shocks)
print("AR(1) simulation:", x)


# Experiment 2: Monte Carlo simulation
last_values = []

# Generate 10,000 independent paths
for i in range(10000):
    shocks = rng.normal(loc=0, scale=1, size=150).tolist()

    path = simulate_ar(
        phi=[0.6],
        initial_values=[0.0],
        shocks=shocks
    )

    # Retain the terminal observation
    last_values.append(path[-1])


# Calculate empirical moments
empirical_mean = np.mean(last_values)
empirical_variance = np.var(last_values, ddof=1)

print("Empirical mean:", empirical_mean)
print("Empirical variance:", empirical_variance)


# Calculate theoretical stationary moments
theoretical_mean = 0.0
theoretical_variance = 1 / (1 - 0.6**2)


# Calculate absolute errors
mean_error = abs(empirical_mean - theoretical_mean)
variance_error = abs(empirical_variance - theoretical_variance)


# Display validation results
print("Theoretical mean:", theoretical_mean)
print("Theoretical variance:", theoretical_variance)

print("Mean error:", mean_error)
print("Variance error:", variance_error)

# Calculate the standard error of the mean
mean_se = np.sqrt(theoretical_variance) / np.sqrt(len(last_values))

# Define the statistical tolerance
mean_tolerance = 3 * mean_se

# Validate the empirical mean
mean_passed = mean_error <= mean_tolerance

print("Mean standard error:", mean_se)
print("Mean tolerance:", mean_tolerance)
print("Mean validation passed:", mean_passed)

# Calculate the standard error of the variance
variance_se = theoretical_variance * np.sqrt(
    2 / (len(last_values) - 1)
)

# Define the statistical tolerance
variance_tolerance = 3 * variance_se

# Validate the empirical variance
variance_passed = variance_error <= variance_tolerance

# Display variance validation results
print("Variance standard error:", variance_se)
print("Variance tolerance:", variance_tolerance)
print("Variance validation passed:", variance_passed)


# Convert terminal observations to a NumPy array
values = np.array(last_values)

# Standardize the observations
z = (values - np.mean(values)) / np.std(values)

# Calculate empirical skewness and kurtosis
empirical_skewness = np.mean(z**3)
empirical_kurtosis = np.mean(z**4)

# Display higher-order moments
print("Empirical skewness:", empirical_skewness)
print("Empirical kurtosis:", empirical_kurtosis)



# Define theoretical higher-order moments
theoretical_skewness = 0.0
theoretical_kurtosis = 3.0

# Calculate absolute errors
skewness_error = abs(empirical_skewness - theoretical_skewness)
kurtosis_error = abs(empirical_kurtosis - theoretical_kurtosis)

# Calculate standard errors
skewness_se = np.sqrt(6 / len(last_values))
kurtosis_se = np.sqrt(24 / len(last_values))

# Define statistical tolerances
skewness_tolerance = 3 * skewness_se
kurtosis_tolerance = 3 * kurtosis_se

# Validate higher-order moments
skewness_passed = skewness_error <= skewness_tolerance
kurtosis_passed = kurtosis_error <= kurtosis_tolerance

# Display validation results
print("Skewness error:", skewness_error)
print("Skewness standard error:", skewness_se)
print("Skewness tolerance:", skewness_tolerance)
print("Skewness validation passed:", skewness_passed)

print("Kurtosis error:", kurtosis_error)
print("Kurtosis standard error:", kurtosis_se)
print("Kurtosis tolerance:", kurtosis_tolerance)
print("Kurtosis validation passed:", kurtosis_passed)
