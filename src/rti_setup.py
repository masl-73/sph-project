import numpy as np

# Simulation Constants
DOMAIN_WIDTH = 0.1
DOMAIN_HEIGHT = 0.05
PARTICLE_SPACING = 0.00028
JITTER_FACTOR = 0.3
DENSITY_LIGHT = 1000.0
DENSITY_HEAVY = 3000.0
INTERFACE_LEVEL = 0.0333333333  # Y-position: Light (bottom 2/3), Heavy (top 1/3)

def setup_rayleigh_taylor(h):
    """
    Sets up the initial conditions for the Rayleigh-Taylor Instability.
    
    Parameters:
    h (float): Smoothing length.
    
    Returns:
    tuple: (positions, velocities, masses, init_densities, colors, rho_refs)
    """
    dx = PARTICLE_SPACING
    dy = dx * np.sqrt(3)/2  # Hexagonal packing vertical spacing
    
    nx = int(DOMAIN_WIDTH / dx)
    ny = int(DOMAIN_HEIGHT / dy)
    
    positions = []
    for i in range(nx):
        for j in range(ny):
            px = i * dx + (0.5 * dx if j % 2 == 1 else 0)
            py = j * dy
            if px < DOMAIN_WIDTH and py < DOMAIN_HEIGHT:
                positions.append([px, py])
                
    positions = np.array(positions)
    
    # Random perturbation to break symmetry
    # jitter = JITTER_FACTOR * dx
    # positions += np.random.uniform(-jitter, jitter, positions.shape)
    
    # Clamp to domain
    positions[:, 0] = np.clip(positions[:, 0], 0, DOMAIN_WIDTH)
    positions[:, 1] = np.clip(positions[:, 1], 0, DOMAIN_HEIGHT)
    
    # Area-based mass calculation
    area_per_particle = dx * dy
    m_light = DENSITY_LIGHT * area_per_particle
    m_heavy = DENSITY_HEAVY * area_per_particle
    
    # Initial Fluid Configuration
    # Heavy fluid on top (y > INTERFACE_LEVEL), Light fluid on bottom
    colors = np.where(positions[:, 1] > INTERFACE_LEVEL, 1, 0)
    
    masses = np.where(colors == 1, m_heavy, m_light)
    init_densities = np.where(colors == 1, DENSITY_HEAVY, DENSITY_LIGHT)
    rho_refs = np.where(colors == 1, DENSITY_HEAVY, DENSITY_LIGHT)
    
    velocities = np.zeros_like(positions)
    
    return positions, velocities, masses, init_densities, colors, rho_refs

def get_domain_size():
    """Returns the domain bounds (min, max)."""
    return np.array([0.0, 0.0]), np.array([DOMAIN_WIDTH, DOMAIN_HEIGHT])
