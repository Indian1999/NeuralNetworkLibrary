import numpy as np

def mse(y_real, y_predicted):
    return np.mean((y_real - y_predicted)**2)

def mse_derivative(y_real, y_predicted):
    return 2 * (y_predicted - y_real) / y_real.size