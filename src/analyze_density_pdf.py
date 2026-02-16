import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def main():
    # 1. Find all checkpoints
    files = sorted(glob.glob("data/checkpoint_*.npz"))
    if not files:
        print("No checkpoints found in data/")
        return

    # Select late-stage frame (e.g. last one) to see evolved state
    last_file = files[-1]
    print(f"Analyzing density PDF for {last_file}...")
    
    data = np.load(last_file)
    densities = data['densities']
    step = data['step']

    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    # Plot Histogram
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Target densities
    rho_light = 1000
    rho_heavy = 2333
    
    # Histogram
    plt.hist(densities, bins=200, color='cyan', alpha=0.7, density=True, label='Particle Density PDF')
    
    # Vertical lines for targets
    plt.axvline(rho_light, color='white', linestyle='--', linewidth=2, label=r'Target $\rho_L=1000$')
    plt.axvline(rho_heavy, color='magenta', linestyle='--', linewidth=2, label=r'Target $\rho_H=2333$')
    
    plt.xlabel('Density (kg/m^3)', color='white')
    plt.ylabel('Probability Density', color='white')
    plt.title(f'Density Distribution at Step {step}', color='white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.grid(True, color='gray', alpha=0.3)
    
    legend = plt.legend(facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')
    
    plt.tight_layout()
    output_file = "output_analysis/density_pdf.png"
    plt.savefig(output_file, facecolor='black', edgecolor='none')
    print(f"Density PDF analysis saved to {output_file}")
    
    # Stats
    std_dev = np.std(densities)
    mean_val = np.mean(densities)
    print(f"Mean Density: {mean_val:.2f}, Std Dev: {std_dev:.2f}")

if __name__ == "__main__":
    main()
