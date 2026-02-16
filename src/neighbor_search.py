import numpy as np
from numba import njit

@njit
def get_cell_id(pos, domain_min, grid_size, n_cells_x, n_cells_y):
    """
    Calculates the 1D grid cell index for a particle position.
    Used for the spatial hashing neighbor search.
    """
    ix = int((pos[0] - domain_min[0]) / grid_size)
    iy = int((pos[1] - domain_min[1]) / grid_size)
    ix = max(0, min(ix, n_cells_x - 1))
    iy = max(0, min(iy, n_cells_y - 1))
    return ix + iy * n_cells_x

def build_grid(positions, domain_min, domain_max, grid_size):
    """
    Builds the spatial hash grid for neighbor searching.
    Returns cell offsets and sorted particle indices.
    """
    n = positions.shape[0]
    n_cells_x = int(np.ceil((domain_max[0] - domain_min[0]) / grid_size)) + 1
    n_cells_y = int(np.ceil((domain_max[1] - domain_min[1]) / grid_size)) + 1
    n_cells = n_cells_x * n_cells_y
    
    cell_ids = np.zeros(n, dtype=np.int32)
    # Numba optimization for loop if possible, but this part mixes numpy and loops.
    # We can keep it simple or wrap the loop in njit if needed.
    # For now, sticking to the previous implementation logic.
    
    # We can use the njit version of get_cell_id in a loop here, or vectorise if possible.
    # The previous implementation had a loop calling get_cell_id.
    
    for i in range(n):
        cell_ids[i] = get_cell_id(positions[i], domain_min, grid_size, n_cells_x, n_cells_y)
        
    sorted_indices = np.argsort(cell_ids).astype(np.int32)
    sorted_cell_ids = cell_ids[sorted_indices]
    
    cell_offsets = np.zeros(n_cells + 1, dtype=np.int32)
    counts = np.bincount(sorted_cell_ids, minlength=n_cells)
    cell_offsets[1:] = np.cumsum(counts)
    
    return n_cells_x, n_cells_y, cell_offsets, sorted_indices
