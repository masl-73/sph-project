import numpy as np
import argparse
import time

from io_utils import save_checkpoint, load_latest_checkpoint, rebuild_history, setup_directories, load_or_init_state
from plot_utils import save_simulation_frame, save_analysis_plots
from rti_setup import get_domain_size
from sph_solver import SPHSolver

# --- Simulation Constants ---
H = 0.0005                 # Smoothing length
DT = 0.000004              # Time step
MAX_STEPS = 80000          # Total simulation steps
P0 = 100000.0              # Reference pressure
GRAVITY = np.array([0, -100.0]) # External gravity
ALPHA = 0.01               # Artificial viscosity coefficient (Reduced from 0.05)
CHECKPOINT_INTERVAL = 300  # Steps between checkpoints
VIZ_INTERVAL = 500         # Steps between visualization frames

# --- Visualization & Analysis ---

def perform_visualization(step, solver, viz_mode, viz_nx=400, viz_ny=200):
    """Generates and saves the main simulation visualization."""
    save_simulation_frame(step, solver, viz_mode, viz_nx, viz_ny)

def perform_analysis(step, t, solver, history):
    """Calculates analysis metrics and updates live plots."""
    pos, vel, mass, dens, col, _, internal_energy = solver.get_state()
    
    # 1. Calculate Metrics
    v_sq = np.sum(vel**2, axis=1)
    Ek = 0.5 * np.sum(mass * v_sq)
    Ep = np.sum(mass * 100.0 * pos[:, 1])
    Etot = Ek + Ep + internal_energy
    
    pos_light = pos[col == 0]
    pos_heavy = pos[col == 1]
    if len(pos_light) > 0 and len(pos_heavy) > 0:
        h_bubble = np.percentile(pos_light[:, 1], 99)
        h_spike = np.percentile(pos_heavy[:, 1], 1)
        width = h_bubble - h_spike
    else:
        width = 0.0
    
    # 2. Update History (avoid duplicates)
    if not history['step'] or history['step'][-1] != step:
        history['step'].append(step)
        history['time'].append(t)
        history['Ek'].append(Ek)
        history['Ep'].append(Ep)
        history['Eint'].append(internal_energy)
        history['Etot'].append(Etot)
        history['mixing_width'].append(width)
        
    # 3. Live Plots
    save_analysis_plots(step, t, pos, vel, history)
    
    print(f"  Rho Avg: {np.mean(dens):.1f} (Min: {np.min(dens):.1f}, Max: {np.max(dens):.1f})", flush=True)
    print(f"  Max Vel: {np.max(np.linalg.norm(vel, axis=1)):.2f}", flush=True)
    print(f"  Energy: Mech={Ek+Ep:.4e}, Int={internal_energy:.4e}, Tot={Etot:.4e}", flush=True)

# --- Simulation Orchestration ---

def print_simulation_config():
    """Prints the current simulation configuration."""
    print("="*40)
    print("SPH Rayleigh-Taylor Simulation Config")
    print("="*40)
    print(f"Smoothing Length (H)  : {H}")
    print(f"Time Step (DT)        : {DT}")
    print(f"Max Steps             : {MAX_STEPS}")
    print(f"Ref Pressure (P0)     : {P0}")
    print(f"Gravity               : {GRAVITY}")
    print(f"Viscosity (Alpha)     : {ALPHA}")
    print(f"Checkpoint Interval   : {CHECKPOINT_INTERVAL}")
    print(f"Visualization Interval: {VIZ_INTERVAL}")
    print("="*40)

def run_simulation(viz_mode='smooth', clear=False):
    """
    Main entry point for the simulation.
    Orchestrates setup, solver initialization, and the main loop.
    """
    # 1. Setup
    print_simulation_config()
    setup_directories(clear)
    
    domain_min, domain_max = get_domain_size()

    # 2. State
    # Note: We pass H to init_state for particle setup
    start_step, pos, vel, mass, dens, col, refs, internal_energy = load_or_init_state(H)
    
    # 3. Solver
    solver = SPHSolver(pos, vel, mass, dens, col, refs,
                       H, DT, P0, GRAVITY, domain_min, domain_max, alpha=ALPHA)
    solver.internal_energy = internal_energy # Set initial internal energy
    print(f"Total particles: {solver.n}")
    
    analysis_history = rebuild_history()

    # 4. Main Loop
    _run_simulation_loop(solver, start_step, MAX_STEPS, DT, viz_mode, analysis_history)

def _run_simulation_loop(solver, start_step, max_steps, dt, viz_mode, history):
    """Execution loop handling physics stepping, IO, and analysis."""
    for step in range(start_step, max_steps):
        solver.step_physics()
        
        # Checkpointing
        if step % CHECKPOINT_INTERVAL == 0:
            pos, vel, mass, dens, col, ref, internal_energy = solver.get_state()
            save_checkpoint(step, pos, vel, mass, dens, col, ref, internal_energy=internal_energy)
            
        # Visualization & Analysis
        if step % VIZ_INTERVAL == 0:
            current_t = step * dt
            print(f"Step {step} (t={current_t:.4f}s)", flush=True)
            perform_visualization(step, solver, viz_mode)
            perform_analysis(step, current_t, solver, history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Rayleigh-Taylor Simulation')
    parser.add_argument('--clear', action='store_true', help='Clear existing checkpoints and start fresh')
    parser.add_argument('--viz-mode', choices=['smooth', 'particles'], default='smooth', help='Visualization style')
    args = parser.parse_args()
    
    run_simulation(viz_mode=args.viz_mode, clear=args.clear)
