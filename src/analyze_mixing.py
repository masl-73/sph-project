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

    print(f"Found {len(files)} checkpoints. Calculating mixing width...")

    steps = []
    times = []
    widths = []

    dt = 0.000004 
    
    # Fluid parameters for Atwood number
    # Try to load from first checkpoint if available
    first_data = np.load(files[0])
    if 'rho_refs' in first_data:
        unique_rhos = np.unique(first_data['rho_refs'])
        if len(unique_rhos) >= 2:
            rho_light = unique_rhos[0]
            rho_heavy = unique_rhos[1]
        else:
            rho_light = 1000.0
            rho_heavy = 3000.0
    else:
        rho_light = 1000.0
        rho_heavy = 3000.0

    Atwood = (rho_heavy - rho_light) / (rho_heavy + rho_light)
    g = 100.0

    print(f"Using Densities: L={rho_light:.0f}, H={rho_heavy:.0f}")
    print(f"Atwood Number A = {Atwood:.3f}")

    for i, filename in enumerate(files):
        data = np.load(filename)
        
        step = data['step']
        positions = data['positions']
        colors = data['colors'] # 0 for light, 1 for heavy
        
        t = step * dt
        
        # Separate fluids
        pos_light = positions[colors == 0]
        pos_heavy = positions[colors == 1]
        
        if len(pos_light) == 0 or len(pos_heavy) == 0:
            continue
            
        # h_bubble: Top of light fluid (rising) -> Use 99th percentile
        h_bubble = np.percentile(pos_light[:, 1], 99)
        
        # h_spike: Bottom of heavy fluid (falling) -> Use 1st percentile
        h_spike = np.percentile(pos_heavy[:, 1], 1)
        
        width = h_bubble - h_spike
        
        steps.append(step)
        times.append(t)
        widths.append(width)

    times = np.array(times)
    widths = np.array(widths)
    
    # 2. Fit Curve (W ~ C * t^2)
    # Filter data to only include growth phase (before hitting boundaries at 0.05m)
    threshold = 0.048
    growth_mask = widths < threshold
    
    fit_times = times[growth_mask]
    fit_widths = widths[growth_mask]
    
    if len(fit_times) > 1:
        t_sq_fit = fit_times**2
        slope, intercept = np.polyfit(t_sq_fit, fit_widths, 1)
        
        alpha_fit = slope / (Atwood * g)
        print(f"Fitted Alpha (Growth Phase): {alpha_fit:.4f}")
        
        # Calculate fitted curve for all times to show the theoretical trajectory
        fitted_curve = slope * (times**2) + intercept
    else:
        alpha_fit = 0
        fitted_curve = np.zeros_like(times)

    # Plotting
    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.plot(times, widths, label='Mixing Width ($h_b - h_s$)', color='cyan', linewidth=4)
    
    if alpha_fit > 0:
        plt.plot(times, fitted_curve, label=r'Fit: $\alpha A g t^2$ ($\alpha \approx ' + f'{alpha_fit:.3f}$)', 
                 color='magenta', linestyle='--', linewidth=2)

    plt.xlabel('Time (s)', color='white')
    plt.ylabel('Width (m)', color='white')
    plt.title(f'Mixing Layer Growth (A={Atwood:.2f})', color='white')
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.grid(True, color='gray', alpha=0.3)
    
    legend = plt.legend(facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')
    
    plt.tight_layout()
    output_file = "output_analysis/mixing_width.png"
    plt.savefig(output_file, facecolor='black', edgecolor='none')
    plt.savefig("mixing_latest.png", facecolor='black', edgecolor='none')
    print(f"Mixing analysis saved to {output_file} and mixing_latest.png")

if __name__ == "__main__":
    main()
