import numpy as np

def generate_ghost_particles(positions, velocities, masses, densities, colors, rho_refs, h, domain_min, domain_max):
    """
    Generates 'Ghost Particles' to enforce slip boundary conditions at the walls.
    Mirrors particles across the boundary with reversed velocity.
    """
    ghost_pos = []
    ghost_vel = []
    ghost_mass = []
    ghost_density = []
    ghost_color = []
    ghost_rho_ref = []
    
    search_dist = 2.0 * h
    
    # Mirroring walls (Left/Right, Bottom/Top)
    for dim, dmin, dmax in [(0, domain_min[0], domain_max[0]), (1, domain_min[1], domain_max[1])]:
        # Lower Boundary
        mask = positions[:, dim] < dmin + search_dist
        if np.any(mask):
            p = positions[mask].copy(); p[:, dim] = 2.0 * dmin - p[:, dim]; ghost_pos.append(p)
            v = velocities[mask].copy(); v[:, dim] = -v[:, dim]; ghost_vel.append(v)
            ghost_mass.append(masses[mask]); ghost_density.append(densities[mask])
            ghost_color.append(colors[mask]); ghost_rho_ref.append(rho_refs[mask])
        
        # Upper Boundary
        mask = positions[:, dim] > dmax - search_dist
        if np.any(mask):
            p = positions[mask].copy(); p[:, dim] = 2.0 * dmax - p[:, dim]; ghost_pos.append(p)
            v = velocities[mask].copy(); v[:, dim] = -v[:, dim]; ghost_vel.append(v)
            ghost_mass.append(masses[mask]); ghost_density.append(densities[mask])
            ghost_color.append(colors[mask]); ghost_rho_ref.append(rho_refs[mask])

    if not ghost_pos:
        return positions, velocities, masses, densities, colors, rho_refs

    all_pos = np.vstack([positions] + ghost_pos)
    all_vel = np.vstack([velocities] + ghost_vel)
    all_mass = np.concatenate([masses] + ghost_mass)
    all_dens = np.concatenate([densities] + ghost_density)
    all_col = np.concatenate([colors] + ghost_color)
    all_ref = np.concatenate([rho_refs] + ghost_rho_ref)
    
    return all_pos, all_vel, all_mass, all_dens, all_col, all_ref
