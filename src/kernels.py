import numpy as np
from numba import njit

# Normalization prefactor for 2D Cubic Spline
KERNEL_PREFACTOR_2D = 10.0 / (7.0 * np.pi)

@njit
def cubic_spline_kernel(r, h):
    """
    Computes the cubic spline kernel value W(r, h).
    This function represents the spatial weighting of particle interactions.
    
    Parameters:
    r (float): Distance between particles.
    h (float): Smoothing length.
    
    Returns:
    float: Kernel value.
    """
    q = r / h
    sigma = KERNEL_PREFACTOR_2D / (h**2)
    
    if q < 1:
        return sigma * (1.0 - 1.5 * q**2 + 0.75 * q**3)
    elif q < 2:
        return sigma * 0.25 * (2.0 - q)**3
    return 0.0

@njit
def cubic_spline_kernel_grad(r_vec, h):
    """
    Computes the gradient of the cubic spline kernel \nabla W(r, h).
    Used for calculating forces (pressure gradient, viscosity).
    
    Parameters:
    r_vec (np.array): Vector pointing from particle j to i (r_ij).
    h (float): Smoothing length.
    
    Returns:
    np.array: Gradient vector.
    """
    r = np.sqrt(r_vec[0]**2 + r_vec[1]**2)
    q = r / h
    sigma = KERNEL_PREFACTOR_2D / (h**2)
    
    if r > 1e-12:
        direction = r_vec / r
    else:
        return np.zeros(2)
    
    if q < 1:
        val = sigma * (1.0/h) * (-3.0*q + 2.25*q**2)
        return val * direction
    elif q < 2:
        val = sigma * (1.0/h) * (-0.75 * (2.0 - q)**2)
        return val * direction
    return np.zeros(2)
