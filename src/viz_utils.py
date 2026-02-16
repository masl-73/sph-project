import numpy as np
from numba import njit, prange
from kernels import KERNEL_PREFACTOR_2D

@njit(parallel=True)
def render_fluid_grid(positions, values, h, nx, ny, domain_x, domain_y):
    grid_val = np.zeros((ny, nx), dtype=np.float32)
    grid_weight = np.zeros((ny, nx), dtype=np.float32)
    
    dx = domain_x / nx
    dy = domain_y / ny
    
    # Kernel radius
    kernel_radius = 2 * h
    
    # Grid search range
    search_cells_x = int(np.ceil(kernel_radius / dx))
    search_cells_y = int(np.ceil(kernel_radius / dy))
    
    # Use 2D Cubic Spline normalization
    sigma = KERNEL_PREFACTOR_2D / (h**2)
    
    for i in prange(len(positions)):
        px, py = positions[i]
        val = values[i]
        
        # Grid index
        idx_x = int(px / dx)
        idx_y = int(py / dy)
        
        # Loop over neighborhood
        for ix in range(max(0, idx_x - search_cells_x), min(nx, idx_x + search_cells_x + 1)):
            for iy in range(max(0, idx_y - search_cells_y), min(ny, idx_y + search_cells_y + 1)):
                # Grid cell center
                gx = (ix + 0.5) * dx
                gy = (iy + 0.5) * dy
                
                dist_sq = (px - gx)**2 + (py - gy)**2
                
                if dist_sq < kernel_radius**2:
                    dist = np.sqrt(dist_sq)
                    q = dist / h
                    # Cubic spline kernel
                    if q < 1.0:
                        w = sigma * (1.0 - 1.5 * q**2 + 0.75 * q**3)
                    elif q < 2.0:
                        w = sigma * 0.25 * (2.0 - q)**3
                    else:
                        w = 0.0
                    
                    if w > 0:
                        grid_val[iy, ix] += val * w
                        grid_weight[iy, ix] += w
                    
    # Normalize (Shepard interpolation)
    for iy in range(ny):
        for ix in range(nx):
            if grid_weight[iy, ix] > 0:
                grid_val[iy, ix] /= grid_weight[iy, ix]
                
    return grid_val
