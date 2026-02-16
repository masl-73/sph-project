import numpy as np
from physics import compute_density, compute_pressure, compute_forces
from boundaries import generate_ghost_particles
from neighbor_search import build_grid

class SPHSolver:
    def __init__(self, positions, velocities, masses, densities, colors, rho_refs, 
                 h, dt, p0, gravity, domain_min, domain_max, alpha=0.02):
        self.positions = positions
        self.velocities = velocities.astype(np.float64)
        self.masses = masses.astype(np.float64)
        self.densities = densities.astype(np.float64)
        self.internal_energy = 0.0 # Total dissipated energy
        
        # Fluid properties
        self.colors = colors
        self.rho_refs = rho_refs
        
        self.h = h
        self.dt = dt
        self.p0 = p0
        self.gravity = gravity
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.alpha = alpha
        
        self.grid_size = 2 * h
        self.n = len(positions)
        
        # Initial Force Calc
        self.step_physics(first_step=True)

    def step_physics(self, first_step=False):
        """
        Performs one time step of the simulation.
        If first_step is True, only computes initial accelerations.
        Otherwise, performs Kick-Drift-Kick integration.
        """
        
        if first_step:
            # Just compute initial accelerations
            self.accelerations, _ = self._compute_accel(self.positions, self.velocities)
            return

        # --- KICK (Half Step Velocity) ---
        v_half = self.velocities + self.accelerations * (self.dt / 2.0)
        
        # --- DRIFT (Full Step Position) ---
        self.positions += v_half * self.dt
        
        # --- BOUNDARY CONDITIONS ---
        for dim, (dmin, dmax) in enumerate(zip(self.domain_min, self.domain_max)):
            mask_lo = self.positions[:, dim] < dmin
            mask_hi = self.positions[:, dim] > dmax
            self.positions[mask_lo, dim] = dmin + (dmin - self.positions[mask_lo, dim])
            self.positions[mask_hi, dim] = dmax - (self.positions[mask_hi, dim] - dmax)
            v_half[mask_lo | mask_hi, dim] *= -0.5
            
        # --- COMPUTE FORCES        # 2. Compute accelerations and force
        new_accelerations, visc_power = self._compute_accel(self.positions, v_half)
        
        # Update internal energy (Dissipated by viscosity)
        # Power is Watts (J/s), so Energy = Power * dt
        self.internal_energy += np.sum(visc_power) * self.dt
        
        # 3. Full step velocity
        self.velocities = v_half + 0.5 * new_accelerations * self.dt
        self.accelerations = new_accelerations

    def _compute_accel(self, pos, vel):
        """
        Internal wrapper for physics kernels.
        Handles Ghost Particles -> Grid Build -> Density -> Pressure -> Forces.
        """
        # 1. Ghost Particles
        all_pos, all_vel, all_mass, all_dens, all_col, all_ref = generate_ghost_particles(
            pos, vel, self.masses, self.densities, self.colors, self.rho_refs, 
            self.h, self.domain_min, self.domain_max
        )
        
        # 2. Build Grid
        nx_cells, ny_cells, cell_offsets, sorted_indices = build_grid(
            all_pos, self.domain_min, self.domain_max, self.grid_size
        )
        
        # 3. Density
        densities_all = compute_density(
            all_pos, all_mass, self.h, self.domain_min, self.grid_size, 
            nx_cells, ny_cells, cell_offsets, sorted_indices
        )
        
        # Update self.densities (real particles only)
        self.densities = densities_all[:self.n]
        
        # 4. Pressure
        pressures_all = compute_pressure(densities_all, all_ref, self.p0)
        
        # Compute Forces
        # Note: we pass n_active to only iterate over real particles
        accels_all, visc_power_all = compute_forces(
            all_pos, all_vel, densities_all, pressures_all, all_mass,
            self.h, self.gravity, self.domain_min, self.grid_size,
            nx_cells, ny_cells, cell_offsets, sorted_indices, self.n,
            alpha=self.alpha
        )
        
        return accels_all, visc_power_all

    def get_state(self):
        """Returns the current state of real particles."""
        return (self.positions, self.velocities, self.masses, 
                self.densities, self.colors, self.rho_refs, self.internal_energy)
