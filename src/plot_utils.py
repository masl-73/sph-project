import matplotlib.pyplot as plt
import numpy as np
from viz_utils import render_fluid_grid

def save_simulation_frame(step, solver, viz_mode, viz_nx=400, viz_ny=200, output_dir='output'):
    """Generates and saves the main simulation visualization."""
    pos, _, _, _, col, _, _ = solver.get_state()
    h = solver.h
    
    plt.figure(figsize=(12, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    if viz_mode == 'smooth':
        grid_img = render_fluid_grid(pos, col, h, viz_nx, viz_ny, 0.1, 0.05)
        plt.imshow(grid_img, origin='lower', extent=[0, 0.1, 0, 0.05], 
                   cmap='cool', vmin=0, vmax=1, interpolation='bicubic')
        plt.title(f"Neon RTI Smooth Step {step}", color='white')
    else:
        plt.scatter(pos[col==0, 0], pos[col==0, 1], c='#00FFFF', s=0.3, edgecolors='none', label='Light')
        plt.scatter(pos[col==1, 0], pos[col==1, 1], c='#FF00FF', s=0.3, edgecolors='none', label='Heavy')
        plt.xlim(0, 0.1); plt.ylim(0, 0.05); plt.gca().set_aspect('equal')
        plt.title(f"Neon RTI Particles Step {step}", color='white')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/step_{step:05d}.png", facecolor='black', edgecolor='none')
    plt.close()

def save_analysis_plots(step, t, pos, vel, history, output_dir='output'):
    """Generates and saves analysis plots (Velocity, Energy, Mixing)."""
    # Velocity Map
    plt.figure(figsize=(10, 5), facecolor='black')
    ax = plt.gca(); ax.set_facecolor('black')
    vel_mag = np.linalg.norm(vel, axis=1)
    scatter = plt.scatter(pos[:, 0], pos[:, 1], c=vel_mag, cmap='inferno', s=0.5, alpha=0.8)
    plt.colorbar(scatter, label='Velocity').ax.tick_params(colors='white')
    plt.title(f"Velocity Map Step {step}", color='white')
    plt.xlim(0, 0.1); plt.ylim(0, 0.05); plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vel_step_{step:05d}.png", facecolor='black', edgecolor='none')
    plt.close()

    # Energy Plot
    plt.figure(figsize=(8, 4), facecolor='black')
    ax = plt.gca(); ax.set_facecolor('black')
    plt.plot(history['time'], history['Ek'], 'cyan', label='Ek')
    plt.plot(history['time'], history['Ep'], 'magenta', label='Ep')
    if 'Eint' in history:
        plt.plot(history['time'], history['Eint'], 'orange', label='Eint (Heat)')
    plt.plot(history['time'], history['Etot'], 'white', linestyle='--', label='Etot')
    plt.legend(); plt.title(f"Energy History (t={t:.3f}s)", color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/energy_latest.png", facecolor='black', edgecolor='none')
    plt.close()

    # Mixing Plot
    plt.figure(figsize=(8, 4), facecolor='black')
    ax = plt.gca(); ax.set_facecolor('black')
    plt.plot(history['time'], history['mixing_width'], 'lime', label='Width')
    plt.legend(); plt.title(f"Mixing Width (t={t:.3f}s)", color='white')
    ax.tick_params(colors='white')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mixing_latest.png", facecolor='black', edgecolor='none')
    plt.close()
