import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# Add src to path to import simulation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.viz_utils import render_fluid_grid
except ImportError:
    try:
        from viz_utils import render_fluid_grid
    except ImportError:
        print("Could not import render_fluid_grid from viz_utils.py")
        sys.exit(1)

def main():
    # 1. Find all checkpoints
    files = sorted(glob.glob("data/checkpoint_*.npz"))
    if not files:
        print("No checkpoints found in data/")
        return

    print(f"Found {len(files)} checkpoints. Calculating vorticity...")

    steps = []
    times = []
    enstrophies = []
    
    dt = 0.000004
    h = 0.0005
    
    # Grid settings (match visualization)
    viz_nx = 400
    viz_ny = 200
    domain_x = 0.1
    domain_y = 0.05
    dx = domain_x / viz_nx
    dy = domain_y / viz_ny

    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    # Process ALL frames for animation
    for i, filename in enumerate(files):
        data = np.load(filename)
        step = data['step']
        positions = data['positions']
        velocities = data['velocities']
        
        t = step * dt
        
        # Interpolate Vx and Vy
        grid_vx = render_fluid_grid(positions, velocities[:, 0], h, viz_nx, viz_ny, domain_x, domain_y)
        grid_vy = render_fluid_grid(positions, velocities[:, 1], h, viz_nx, viz_ny, domain_x, domain_y)
        
        # Compute Gradients: dVy/dx - dVx/dy
        grad_vy = np.gradient(grid_vy, dy, dx)
        grad_vx = np.gradient(grid_vx, dy, dx)
        
        dvy_dx = grad_vy[1]
        dvx_dy = grad_vx[0]
        vorticity = dvy_dx - dvx_dy
        
        enstrophy = np.sum(vorticity**2) * (dx * dy)
        
        steps.append(step)
        times.append(t)
        enstrophies.append(enstrophy)
        
        # Save Vorticity Map for every frame
        print(f"Generating Vorticity Map for step {step} ({i+1}/{len(files)})...")
        plt.figure(figsize=(10, 5), facecolor='black')
        ax = plt.gca()
        ax.set_facecolor('black')
        
        # Use a fixed scale for animation stability
        # Based on previous runs, +/- 50 to 100 is a good range for vorticity
        limit = 100.0 
        im = plt.imshow(vorticity, origin='lower', extent=[0, domain_x, 0, domain_y], 
                   cmap='seismic', vmin=-limit, vmax=limit, interpolation='bicubic')
        
        plt.axis('off')
        plt.title(f"Vorticity Field (Step {step})", color='white')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Vorticity (1/s)', color='white')
        
        output_map = f"output_analysis/vorticity_map_{step:05d}.png"
        plt.tight_layout()
        plt.savefig(output_map, facecolor='black', edgecolor='none')
        plt.close()

    # Plot Enstrophy Evolution
    times = np.array(times)
    enstrophies = np.array(enstrophies)
    
    plt.figure(figsize=(10, 5), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.plot(times, enstrophies, color='yellow', linewidth=2, label='Enstrophy')
    
    plt.xlabel('Time (s)', color='white')
    plt.ylabel('Enstrophy', color='white')
    plt.title('Enstrophy Evolution (Turbulence Intensity)', color='white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.grid(True, color='gray', alpha=0.3)
    
    plt.tight_layout()
    output_enstrophy = "output_analysis/enstrophy.png"
    plt.savefig(output_enstrophy, facecolor='black', edgecolor='none')
    print(f"Enstrophy analysis saved to {output_enstrophy}")

if __name__ == "__main__":
    main()
