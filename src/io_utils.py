import numpy as np
import os
import glob
import json
import shutil
from rti_setup import setup_rayleigh_taylor

def setup_directories(clear=False):
    """Prepares the data and output directories."""
    if clear:
        if os.path.exists('data'):
            print("Clearing existing checkpoints...")
            shutil.rmtree('data')
        if os.path.exists('output'):
            print("Clearing existing output images...")
            shutil.rmtree('output')
            
    if not os.path.exists('output'):
        os.makedirs('output')
    # data/ is created by save_checkpoint, but good to ensure
    if not os.path.exists('data'):
        os.makedirs('data')

def save_checkpoint(step, positions, velocities, masses, densities, colors, rho_refs, internal_energy=0.0, data_dir='data'):
    """Saves the simulation state to a compressed .npz file."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    filename = os.path.join(data_dir, f"checkpoint_{step:05d}.npz")
    np.savez(filename, step=step, positions=positions, velocities=velocities, 
             masses=masses, densities=densities, colors=colors, rho_refs=rho_refs,
             internal_energy=internal_energy)
    print(f"Checkpoint saved: {filename}", flush=True)

def load_latest_checkpoint(data_dir='data'):
    """Loads the most recent checkpoint from the data directory."""
    if not os.path.exists(data_dir):
        return None
    files = glob.glob(os.path.join(data_dir, "checkpoint_*.npz"))
    if not files:
        return None
    # Sort by step number in filename
    latest_file = max(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    print(f"Loading checkpoint: {latest_file}", flush=True)
    return np.load(latest_file)

def load_or_init_state(h):
    """
    Loads the latest checkpoint or initializes a new simulation state.
    Returns: (start_step, positions, velocities, masses, densities, colors, rho_refs)
    """
    checkpoint = load_latest_checkpoint()
    
    if checkpoint:
        start_step = int(checkpoint['step']) + 1
        # Handle backward compatibility for checkpoints without internal_energy
        internal_energy = float(checkpoint['internal_energy']) if 'internal_energy' in checkpoint else 0.0
        
        print(f"Resuming simulation from step {start_step}", flush=True)
        return (start_step, 
                checkpoint['positions'], checkpoint['velocities'], 
                checkpoint['masses'], checkpoint['densities'], 
                checkpoint['colors'], checkpoint['rho_refs'],
                internal_energy)
    else:
        print("Starting new simulation")
        start_step = 0
        internal_energy = 0.0
        pos, vel, mass, dens, col, refs = setup_rayleigh_taylor(h)
        return start_step, pos, vel, mass, dens, col, refs, internal_energy

def rebuild_history(data_dir='data'):
    """Reconstructs analysis history from existing checkpoint files."""
    history = {
        'step': [], 'time': [],
        'Ek': [], 'Ep': [], 'Eint': [], 'Etot': [],
        'mixing_width': []
    }
    
    files = sorted(glob.glob(os.path.join(data_dir, "checkpoint_*.npz")))
    if not files:
        return history
        
    print(f"Rebuilding history from {len(files)} checkpoints...")
    
    dt = 0.000004
    g = 100.0 # Standard gravity for potential energy calc
    
    for i, filename in enumerate(files):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(files)} checkpoints...", flush=True)
        try:
            data = np.load(filename)
            step = int(data['step'])
            positions = data['positions']
            velocities = data['velocities']
            masses = data['masses']
            colors = data['colors']
            
            # Load internal energy if available
            E_int = float(data['internal_energy']) if 'internal_energy' in data else 0.0
            
            t = step * dt
            
            # Energy Calculation
            v_sq = np.sum(velocities**2, axis=1)
            Ek = 0.5 * np.sum(masses * v_sq)
            Ep = np.sum(masses * g * positions[:, 1])
            Etot = Ek + Ep + E_int # Total Energy including dissipated heat
            
            # Mixing Width Calculation
            pos_light = positions[colors == 0]
            pos_heavy = positions[colors == 1]
            
            if len(pos_light) > 0 and len(pos_heavy) > 0:
                h_bubble = np.percentile(pos_light[:, 1], 99)
                h_spike = np.percentile(pos_heavy[:, 1], 1)
                width = h_bubble - h_spike
            else:
                width = 0.0
                
            history['step'].append(step)
            history['time'].append(t)
            history['Ek'].append(Ek)
            history['Ep'].append(Ep)
            history['Eint'].append(E_int)
            history['Etot'].append(Etot)
            history['mixing_width'].append(width)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return history
