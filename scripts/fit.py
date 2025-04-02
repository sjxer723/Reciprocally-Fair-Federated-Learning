import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def curve_function(x, alpha, beta):
    return 1 - alpha * x**beta


size = [10, 20, 40, 80, 160]
accs_cifar10_first50 = [0.2994, 0.3362, 0.4997, 0.5911, 0.6875]
accs_cifar10_last50 = [0.3108, 0.3753, 0.4612, 0.5961, 0.6524]
initial_guess = [0.2, 1.5]
fit_params, _ = curve_fit(curve_function, size, accs_cifar10_first50, p0=initial_guess)
alpha_fit, beta_fit = fit_params

print("alpha:", alpha_fit)
print("beta:", beta_fit)
