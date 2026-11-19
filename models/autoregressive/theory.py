
import numpy as np

def is_stationary(phi):

    # Construct the characteristic polynomial
    coefficients = [1.0]

    for coefficient in phi:
        coefficients.append(-coefficient)

    coefficients.reverse()

    # Calculate the polynomial roots
    roots = np.roots(coefficients)

    # Check the stationarity condition
    return np.all(np.abs(roots) > 1)



def theoretical_autocovariances(phi, sigma2=1.0):
    # Check stationarity
    if not is_stationary(phi):
        raise ValueError("The AR model is not stationary.")
    p = len(phi)

    # Initialize the system
    A = np.eye(p + 1)
    b = np.zeros(p + 1)

    b[0] = sigma2

    # Construct the Yule-Walker equations
    for k in range(p + 1):
        for j in range(1, p + 1):
            A[k, abs(k - j)] -= phi[j - 1]

    # Solve the system
    gamma = np.linalg.solve(A, b)

    return gamma




