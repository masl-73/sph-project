import numpy as np
from numba import njit, prange
from kernels import cubic_spline_kernel, cubic_spline_kernel_grad, KERNEL_PREFACTOR_2D

# Physical Constants
EOS_GAMMA = 7.0         # Exponent in Tait's Equation of State (7 for water-like fluids)
SOUND_SPEED = 30.0      # Reference speed of sound (must be > 10 * max_velocity)
MIN_DENSITY = 0.1       # Threshold to ignore vacuum/invalid particles
VISCOSITY_EPSILON = 0.01 # Small factor in viscosity to prevent division by zero
KERNEL_CUTOFF_SQ_FACTOR = 4.0 # (2h)^2 = 4h^2
MIN_DIST_SQ = 1e-18     # Minimum squared distance to avoid division by zero
GRID_SEARCH_RADIUS = 1  # Number of cells to search in each direction (Moore neighborhood of rank 1)

@njit
def calculate_pressure_force(p_i, p_j, rho_i, rho_j, m_j, grad_w):
    """
    Calculates the symmetrical pressure gradient force contribution from particle j on i.
    F_p = - m_j * (P_i/rho_i^2 + P_j/rho_j^2) * \nabla W
    """
    p_term = (p_i / rho_i**2 + p_j / rho_j**2)
    return -m_j * p_term * grad_w

@njit
def calculate_viscosity_force(r_vec, v_vec, r2, rho_i, rho_j, m_j, h, alpha, beta, grad_w):
    """
    Calculates the Monaghan artificial viscosity force contribution.
    Dampens shockwaves and stabilizes numerical oscillations.
    """
    # Dot product of velocity difference and position difference
    v_dot_r = v_vec[0] * r_vec[0] + v_vec[1] * r_vec[1]
    
    if v_dot_r < 0:
        # Viscosity is only applied when particles are approaching each other
        mu = (h * v_dot_r) / (r2 + VISCOSITY_EPSILON * h**2)
        rho_bar = (rho_i + rho_j) / 2.0
        
        # Viscosity term \Pi_{ij}
        phi = (-alpha * SOUND_SPEED * mu + beta * mu**2) / rho_bar
        return -m_j * phi * grad_w
        
    return np.zeros(2)

@njit(parallel=True)
def compute_density(positions, masses, h, domain_min, grid_size, n_cells_x, n_cells_y, cell_offsets, sorted_indices):
    """
    Computes the density \rho_i for each particle using SPH summation.
    
    \rho_i = \sum_j m_j W(r_{ij}, h)
    
    Uses a grid-based neighbor search for efficiency.
    """
    n = positions.shape[0]
    densities = np.zeros(n)
    h2 = h**2
    
    for i in prange(n):
        ix = int((positions[i, 0] - domain_min[0]) / grid_size)
        iy = int((positions[i, 1] - domain_min[1]) / grid_size)
        
        d_i = 0.0
        # Iterate over neighboring cells within search radius
        for dx in range(-GRID_SEARCH_RADIUS, GRID_SEARCH_RADIUS + 1):
            for dy in range(-GRID_SEARCH_RADIUS, GRID_SEARCH_RADIUS + 1):
                nx_idx = ix + dx
                ny_idx = iy + dy
                
                # Check grid bounds
                if nx_idx >= 0 and nx_idx < n_cells_x and ny_idx >= 0 and ny_idx < n_cells_y:
                    cell_id = nx_idx + ny_idx * n_cells_x
                    start = cell_offsets[cell_id]
                    end = cell_offsets[cell_id + 1]
                    
                    # Iterate over particles in cell
                    for j_ptr in range(start, end):
                        j = sorted_indices[j_ptr]
                        dx_v = positions[i, 0] - positions[j, 0]
                        dy_v = positions[i, 1] - positions[j, 1]
                        r2 = dx_v**2 + dy_v**2
                        
                        if r2 < KERNEL_CUTOFF_SQ_FACTOR * h2: # Cutoff radius 2h
                            r = np.sqrt(r2)
                            d_i += masses[j] * cubic_spline_kernel(r, h)
        
        densities[i] = d_i
            
    return densities

@njit
def compute_pressure(densities, rho_refs, p0, gamma=EOS_GAMMA):
    """
    Computes pressure using the Tait Equation of State (EOS).
    P_i = P_0 * ((\rho_i / \rho_{ref,i})^\gamma - 1)
    """
    return p0 * ((densities / rho_refs)**gamma - 1)

@njit(parallel=True)
def compute_forces(positions, velocities, densities, pressures, masses, h, gravity, 
                   domain_min, grid_size, n_cells_x, n_cells_y, 
                   cell_offsets, sorted_indices, n_active,
                   alpha=0.1, beta=0.0):
    """
    Computes acceleration for each particle due to:
    1. Pressure Gradient
    2. Artificial Viscosity
    3. External Gravity
    Returns:
    accelerations: (n_active, 2)
    viscosity_power: (n_active,) - Power dissipated by viscosity (W)
    """
    accelerations = np.zeros((n_active, 2))
    viscosity_power = np.zeros(n_active)
    h2 = h**2
    
    for i in prange(n_active):
        # Local accumulation for viscosity to compute work
        acc_visc_x = 0.0
        acc_visc_y = 0.0
        
        # 1. Gravity
        accelerations[i, 0] = gravity[0]
        accelerations[i, 1] = gravity[1]
        
        if np.isnan(densities[i]) or densities[i] < MIN_DENSITY: continue
        
        ix = int((positions[i, 0] - domain_min[0]) / grid_size)
        iy = int((positions[i, 1] - domain_min[1]) / grid_size)
        
        # Neighbor search
        for dx in range(-GRID_SEARCH_RADIUS, GRID_SEARCH_RADIUS + 1):
            for dy in range(-GRID_SEARCH_RADIUS, GRID_SEARCH_RADIUS + 1):
                nx_idx = ix + dx
                ny_idx = iy + dy
                
                if nx_idx >= 0 and nx_idx < n_cells_x and ny_idx >= 0 and ny_idx < n_cells_y:
                    cell_id = nx_idx + ny_idx * n_cells_x
                    start = cell_offsets[cell_id]
                    end = cell_offsets[cell_id + 1]
                    
                    for j_ptr in range(start, end):
                        j = sorted_indices[j_ptr]
                        if i == j: continue # Skip self
                        if np.isnan(densities[j]) or densities[j] < MIN_DENSITY: continue
                        
                        dx_vec = positions[i, 0] - positions[j, 0]
                        dy_vec = positions[i, 1] - positions[j, 1]
                        r2 = dx_vec**2 + dy_vec**2
                        
                        if r2 < KERNEL_CUTOFF_SQ_FACTOR * h2 and r2 > MIN_DIST_SQ:
                            r = np.sqrt(r2)
                            r_v = np.array([dx_vec, dy_vec])
                            grad_w = cubic_spline_kernel_grad(r_v, h)
                            
                            # 2. Pressure Force
                            f_pressure = calculate_pressure_force(
                                pressures[i], pressures[j], densities[i], densities[j], masses[j], grad_w
                            )
                            accelerations[i] += f_pressure
                            
                            # 3. Artificial Viscosity
                            v_vec = np.array([velocities[i, 0] - velocities[j, 0], 
                                              velocities[i, 1] - velocities[j, 1]])
                            
                            f_viscosity = calculate_viscosity_force(
                                r_v, v_vec, r2, densities[i], densities[j], masses[j], h, alpha, beta, grad_w
                            )
                            accelerations[i] += f_viscosity
                            acc_visc_x += f_viscosity[0]
                            acc_visc_y += f_viscosity[1]
                                
                             
        # Compute dissipated power for particle i: P = - F_visc . v_i
        # F_visc = m_i * a_visc
        # We use v_i (which is v_half in the solver call)
        
        # Note: Viscosity acts as friction, so F_visc . v is generally negative (removing energy).
        # We want the positive amount of energy converted to heat (dissipated).
        # Dissipated Power = - (m_i * a_visc . v_i)
        
        v_dot_a = acc_visc_x * velocities[i, 0] + acc_visc_y * velocities[i, 1]
        viscosity_power[i] = -masses[i] * v_dot_a
        
    return accelerations, viscosity_power
