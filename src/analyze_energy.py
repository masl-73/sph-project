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

    print(f"Found {len(files)} checkpoints. Calculating energy...")

    steps = []
    times = []
    Ek_list = []
    Ep_list = []
    Ei_list = []
    Etot_list = []

    dt = 0.000004 # Time step from simulation.py
    g = 100.0     # Gravity magnitude

    for i, filename in enumerate(files):
        if i % 10 == 0:
            print(f"Processing {filename}...")
            
        data = np.load(filename)
        
        step = data['step']
        positions = data['positions']
        velocities = data['velocities']
        masses = data['masses']
        
        # Current time
        t = step * dt
        
        # Kinetic Energy: 0.5 * m * v^2
        # v^2 = vx^2 + vy^2
        v_sq = np.sum(velocities**2, axis=1)
        Ek = 0.5 * np.sum(masses * v_sq)
        
        # Potential Energy: m * g * y
        y_coords = positions[:, 1]
        Ep = np.sum(masses * g * y_coords)
        
        # Internal Energy (Heat/Viscous work)
        Ei = float(data['internal_energy']) if 'internal_energy' in data else 0.0
        
        Etot = Ek + Ep + Ei
        
        steps.append(step)
        times.append(t)
        Ek_list.append(Ek)
        Ep_list.append(Ep)
        Ei_list.append(Ei)
        Etot_list.append(Etot)

    # Convert to numpy arrays for easier plotting
    times = np.array(times)
    Ek_list = np.array(Ek_list)
    Ep_list = np.array(Ep_list)
    Ei_list = np.array(Ei_list)
    Etot_list = np.array(Etot_list)

    # Plotting
    if not os.path.exists('output_analysis'):
        os.makedirs('output_analysis')

    plt.figure(figsize=(10, 6), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    # Plot Energies
    plt.plot(times, Ek_list, label='Kinetic (Ek)', color='cyan', linewidth=4)
    plt.plot(times, Ep_list, label='Potential (Ep)', color='magenta', linewidth=4)
    plt.plot(times, Ei_list, label='Internal (Eint)', color='orange', linewidth=4)
    plt.plot(times, Etot_list, label='Total (Ek+Ep+Eint)', color='white', linewidth=5, linestyle='--')
    
    plt.xlabel('Time (s)', color='white', fontsize=12)
    plt.ylabel('Energy (J)', color='white', fontsize=12)
    plt.title('Energy Balance over Time', color='white', fontsize=16)
    
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Grid
    plt.grid(True, color='gray', alpha=0.3)
    
    # Legend
    legend = plt.legend(facecolor='black', edgecolor='white')
    plt.setp(legend.get_texts(), color='white')
    
    plt.tight_layout()
    output_file = "output_analysis/energy_balance.png"
    plt.savefig(output_file, facecolor='black', edgecolor='none')
    plt.savefig("energy_latest.png", facecolor='black', edgecolor='none')
    print(f"Energy analysis saved to {output_file} and energy_latest.png")

    # Print failure statistics (how much energy changed?)
    E_start = Etot_list[0]
    E_end = Etot_list[-1]
    loss_pct = (E_end - E_start) / E_start * 100
    print(f"Total Energy Change: {loss_pct:.2f}%")

if __name__ == "__main__":
    main()
