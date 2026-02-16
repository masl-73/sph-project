import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# Add src to path to import simulation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from src.simulation import render_fluid_grid
except ImportError:
    try:
        from simulation import render_fluid_grid
    except ImportError:
        print("Could not import render_fluid_grid from simulation.py")
        sys.exit(1)

def main():
    # 1. Find all checkpoints
    files = sorted(glob.glob("data/checkpoint_*.npz"))
    if not files:
        print("No checkpoints found in data/")
        return

    # Select late-stage checkpoints where turbulence is developed
    # e.g., last 10 files
    selected_files = files[-10:] if len(files) > 10 else files
    
    print(f"Analyzing spectra for last {len(selected_files)} checkpoints...")
    
    dt = 0.000004
    h = 0.0005
    
    # Grid settings
    viz_nx = 400
    viz_ny = 200 # Use square grid for easier isotropic spectra? 
    # Actually for FFT, square is better, or we handle aspect ratio carefully.
    # To keep it simple, let's use the existing grid and compute 1D spectrum by averaging.
    
    domain_x = 0.1
    domain_y = 0.05
    
    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    avg_spectrum = None
    k_axis = None

    for i, filename in enumerate(selected_files):
        print(f"Processing {filename}...")
        data = np.load(filename)
        positions = data['positions']
        velocities = data['velocities']
        
        # Interpolate Velocity Components
        grid_vx = render_fluid_grid(positions, velocities[:, 0], h, viz_nx, viz_ny, domain_x, domain_y)
        grid_vy = render_fluid_grid(positions, velocities[:, 1], h, viz_nx, viz_ny, domain_x, domain_y)
        
        # Compute Kinetic Energy Density in Fourier Space
        # FFT
        fft_vx = np.fft.fft2(grid_vx)
        fft_vy = np.fft.fft2(grid_vy)
        
        # Shift zero frequency to center
        fft_vx = np.fft.fftshift(fft_vx)
        fft_vy = np.fft.fftshift(fft_vy)
        
        # Power Spectrum Density (PSD) ~ |F|^2
        psd = 0.5 * (np.abs(fft_vx)**2 + np.abs(fft_vy)**2)
        
        # Compute 1D isotropic spectrum
        # Get wavenumbers
        kx = np.fft.fftshift(np.fft.fftfreq(viz_nx, d=domain_x/viz_nx))
        ky = np.fft.fftshift(np.fft.fftfreq(viz_ny, d=domain_y/viz_ny))
        
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)
        
        # Binning by K magnitude
        k_bins = np.arange(0, np.max(K), 50) # bin width 50
        k_vals = 0.5 * (k_bins[:-1] + k_bins[1:])
        E_k = np.zeros_like(k_vals)
        
        for j in range(len(k_bins)-1):
            mask = (K >= k_bins[j]) & (K < k_bins[j+1])
            if np.any(mask):
                E_k[j] = np.sum(psd[mask])
        
        if avg_spectrum is None:
            avg_spectrum = E_k
            k_axis = k_vals
        else:
            avg_spectrum += E_k

    # Average over selected frames
    avg_spectrum /= len(selected_files)

    # Plot
    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    plt.loglog(k_axis, avg_spectrum, 'cyan', linewidth=2, label='Simulation Spectrum')
    
    # Reference Slopes
    # Try fitting a portion of the inertial range (e.g. middle frequencies)
    # Or just plot reference lines
    
    # k^-3 (2D Enstrophy Cascade)
    ref_k = k_axis[10:50] 
    if len(ref_k) > 0:
        ref_E3 = ref_k**(-3) * (avg_spectrum[20] / ref_k[10]**(-3)) # anchor scaling
        plt.loglog(ref_k, ref_E3, 'magenta', linestyle='--', linewidth=2, label=r'$k^{-3}$ (2D Turbulence)')
        
        # k^-5/3 (3D Energy Cascade / Inverse Cascade)
        ref_E53 = ref_k**(-5/3) * (avg_spectrum[20] / ref_k[10]**(-5/3))
        # plt.loglog(ref_k, ref_E53, 'yellow', linestyle=':', linewidth=2, label=r'$k^{-5/3}$')

    plt.xlabel('Wavenumber k', color='white')
    plt.ylabel('Energy E(k)', color='white')
    plt.title('Kinetic Energy Power Spectrum', color='white')
    
    ax.tick_params(axis='x', colors='white', which='both')
    ax.tick_params(axis='y', colors='white', which='both')
    plt.grid(True, color='gray', alpha=0.3, which='both')
    
    legend = plt.legend(facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')
    
    plt.tight_layout()
    output_file = "output_analysis/power_spectrum.png"
    plt.savefig(output_file, facecolor='black', edgecolor='none')
    print(f"Spectra analysis saved to {output_file}")

if __name__ == "__main__":
    main()
