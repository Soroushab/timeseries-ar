
# Autoregressive Time Series Models

## Overview

A Python implementation of autoregressive models of arbitrary
order, AR(p), developed as part of a research assistantship.

The project provides a computational foundation for the
subsequent development of score-driven time series models.

## Features

- AR(p) simulation
- Stationarity checking
- Yule-Walker equations
- Theoretical autocovariance calculation
- Monte Carlo simulation
- Statistical moment validation
- Automated testing with pytest

## Statistical Methodology

The empirical moments are estimated from the terminal
observations of independent simulated paths.

The following statistics are compared with their
theoretical counterparts:

- Mean
- Variance
- Skewness
- Kurtosis

Validation uses a tolerance of three standard errors,
with asymptotic approximations for higher-order moments.

## Monte Carlo Results

| Model | Seed | Passed criteria |
|-------|------|-----------------|
| AR(2) | 42   | 3/4             |
| AR(2) | 43   | 4/4             |
| AR(3) | 2026 | 4/4             |

The skewness criterion was not satisfied for AR(2)
with seed 42. This result is retained for transparency
and reproducibility.

## Limitations

The validation framework assumes stationary, causal,
zero-intercept Gaussian AR processes with positive
innovation variance.

The default simulation length may be insufficient
for processes close to the stationarity boundary.

Input validation is not exhaustive.

## Dependencies

- Python
- NumPy
- pytest

## Running the Tests

    python -m pytest -q

## Status

AR(p) simulation and preliminary statistical validation
implemented. All eight automated tests passed.
