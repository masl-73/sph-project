import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import argparse

def main():
    # 1. Find all checkpoints
    files = sorted(glob.glob("data/checkpoint_*.npz"))
    if not files:
        print("No checkpoints found in data/")
        return

    total_files = len(files)
    print(f"Found {total_files} checkpoints.")

    # 2. Select 10 evenly spaced checkpoints
    # We want to include the first and last, and 8 in between
    indices = np.linspace(0, total_files - 1, 10, dtype=int)
    selected_files = [files[i] for i in indices]

    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    print(f"Processing {len(selected_files)} files...")

    # Global max velocity for consistent colorbar (optional, but good for comparison)
    # To save time we might compute per-frame, or do a quick pass.
    # Let's do per-frame scaling to see details best in each frame, 
    # but maybe fix x-axis of histogram if needed. For now, dynamic is fine.

    for i, filename in enumerate(selected_files):
        print(f"Processing {filename} ({i+1}/10)...")
        data = np.load(filename)
        
        step = data['step']
        positions = data['positions']
        velocities = data['velocities']
        
        # Calculate velocity magnitude
        vel_mag = np.linalg.norm(velocities, axis=1)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 3a. Spatial Scatter Plot
        # Use a dark background style
        fig.patch.set_facecolor('black')
        ax1.set_facecolor('black')
        
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], c=vel_mag, cmap='inferno', s=0.5, alpha=0.8)
        ax1.set_title(f"Velocity Magnitude Field (Step {step})", color='white')
        ax1.set_aspect('equal')
        ax1.set_xlim(0, 0.1)
        ax1.set_ylim(0, 0.05)
        ax1.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Velocity (m/s)', color='white')

        # 3b. Histogram
        ax2.set_facecolor('#1a1a1a') # Slightly lighter dark for contrast
        ax2.hist(vel_mag, bins=100, color='orange', edgecolor='none', alpha=0.7, log=True)
        ax2.set_title(f"Velocity Distribution (Log Scale)", color='white')
        ax2.set_xlabel("Velocity (m/s)", color='white')
        ax2.set_ylabel("Count (Log)", color='white')
        
        ax2.spines['bottom'].set_color('white')
        ax2.spines['top'].set_color('white')
        ax2.spines['left'].set_color('white')
        ax2.spines['right'].set_color('white')
        ax2.tick_params(axis='x', colors='white')
        ax2.tick_params(axis='y', colors='white')

        plt.suptitle(f"Analysis Step {step}", color='white', fontsize=16)
        
        output_file = f"output_analysis/velocity_step_{step:05d}.png"
        plt.tight_layout()
        plt.savefig(output_file, facecolor='black', edgecolor='none')
        plt.close()
        
    print("Analysis complete. Images saved to output_analysis/")

if __name__ == "__main__":
    main()
