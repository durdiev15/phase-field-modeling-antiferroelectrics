import torch
import itertools
import h5py
import sys
import os

def random_polarization_field(Nx, Ny, freq2):
    # Random periodic polarization
    torch.manual_seed(42)
    rfft_shape = (2, Nx, Ny // 2 + 1)
    P_hat_random = torch.randn(rfft_shape, dtype=torch.complex64)
    smoothness_sigma = (Nx + Ny) / 2.0 / 40.0 
    k_squared_rfft = freq2[:, :Ny//2+1]
    gaussian_filter = torch.exp(-k_squared_rfft / (2 * (smoothness_sigma**2)))
    P_hat_filtered = P_hat_random * gaussian_filter
    P = torch.fft.irfftn(P_hat_filtered, s=(Nx, Ny), dim=(1, 2))
    return P

# ========================= Voigt Tensors ===============================
def full_2x2_to_Voigt_3_index(i, j):
    if i == j:
        return i  # 0 for (0,0), 1 for (1,1)
    return 2 

def stiffness_tensor_2d(C11, C12, C44):
    return torch.tensor([
        [C11, C12, 0],  # xx, yy, xy components
        [C12, C11, 0],  # yy, xx, xy components
        [0,   0,  C44]  # xy, xy (shear)
    ])

def voigt_to_full_tensor_2D(C11, C12, C44):
    C_voigt = stiffness_tensor_2d(C11, C12, C44)
    C_out = torch.zeros((2, 2, 2, 2))
    for i, j, k, l in itertools.product(range(2), repeat=4):
        vi = full_2x2_to_Voigt_3_index(i, j)
        vj = full_2x2_to_Voigt_3_index(k, l)
        C_out[i, j, k, l] = C_voigt[vi, vj]
        
    return C_out

# ================== Initial polarization setup ===================================
def initial_polarization(sim_params, filepath: str = None):

    if filepath: # you may want to start with polarization from a .h5 file
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} does not exist.")
            sys.exit(1)

        with h5py.File(filepath, 'r') as hf:
            sim_params_group = hf['Simulation_Parameters']
            sim_params_file = {key: sim_params_group[key][()] for key in sim_params_group}

            try:
                P1 = torch.tensor(hf[f'Polarization/Px/time_{sim_params_file['nsteps']}'][()])
                P2 = torch.tensor(hf[f'Polarization/Py/time_{sim_params_file['nsteps']}'][()])
            except KeyError as e:
                print(f"Error: {e}. Could not find polarization data at final time step.")
                sys.exit(1)

            return torch.stack([P1, P2]), sim_params_file

    else:
        torch.manual_seed(42) # Random initial polarization between -0.01 and +0.01
        return 0.01 * (2.0 * torch.rand(2, sim_params['Nx'], sim_params['Ny']) - 1.0), sim_params

# =================== Triangular applied electric field for hysteresis =====
def triangular_field_vectorized(t, E_max, T_cycle): # Custom triangular waveform starting from 0
    t_mod = torch.remainder(t, T_cycle)
    quarter = T_cycle / 4

    E = torch.zeros_like(t)

    mask1 = t_mod < quarter
    mask2 = (t_mod >= quarter) & (t_mod < 2 * quarter)
    mask3 = (t_mod >= 2 * quarter) & (t_mod < 3 * quarter)
    mask4 = t_mod >= 3 * quarter

    E[mask1] = E_max * (t_mod[mask1] / quarter)
    E[mask2] = E_max * (1 - (t_mod[mask2] - quarter) / quarter)
    E[mask3] = -E_max * ((t_mod[mask3] - 2 * quarter) / quarter)
    E[mask4] = -E_max * (1 - (t_mod[mask4] - 3 * quarter) / quarter)

    return E
# =============== Derivative of spontaneous strain ===========================
def spon_strain_derivative(Q_11, Q_12, Q_44, P):
    """
    Compute the spontaneous strain tensor and its derivative w.r.t. polarization components in 2D (full tensor form).

    Args:
        Q_11, Q_12, Q_44: Electrostrictive coefficients (scalars)
        P: Polarization field, shape (2, Nx, Ny)

    Returns:
        eps0: Spontaneous strain tensor, shape (2, 2, Nx, Ny)
        deps_dP: Derivative of strain w.r.t. P, shape (2, 2, 2, Nx, Ny)
    """
    P0, P1 = P[0], P[1]
    shape = P0.shape

    # Spontaneous strain tensor (2x2)
    eps0 = torch.empty((2, 2, *shape), dtype=P.dtype, device=P.device)
    eps0[0, 0] = P0**2 * Q_11 + P1**2 * Q_12
    eps0[1, 1] = P0**2 * Q_12 + P1**2 * Q_11
    eps0[0, 1] = eps0[1, 0] = P0 * P1 * Q_44

    # Derivatives: shape (2, 2, 2, Nx, Ny)
    deps_dP = torch.zeros((2, 2, 2) + shape, dtype=P.dtype, device=P.device)

    # deps_ij / dP_k
    # Derivative w.r.t. P0
    deps_dP[0, 0, 0] = 2 * P0 * Q_11
    deps_dP[1, 1, 0] = 2 * P0 * Q_12
    deps_dP[0, 1, 0] = deps_dP[1, 0, 0] = P1 * Q_44

    # Derivative w.r.t. P1
    deps_dP[0, 0, 1] = 2 * P1 * Q_12
    deps_dP[1, 1, 1] = 2 * P1 * Q_11
    deps_dP[0, 1, 1] = deps_dP[1, 0, 1] = P0 * Q_44

    return eps0, deps_dP

def log_mechanics_solver_data(
        filename: str,
        step: int,
        itr_total: int,
        res_data: list
    ):
    """
    Appends mechanics solver log to a .txt file.

    Args:
        filename (str): Path to log file.
        step (int): Global time step number.
        itr_total (int): Number of iterations in the current step.
        res_data (list): List of residuals for each iteration (float or tensor).
    """
    with open(filename, "a") as f:
        f.write(f"#--------------------- Step: {step} --------------------------\n")
        for it in range(itr_total):
            res = res_data[it].item() if hasattr(res_data[it], "item") else res_data[it]
            f.write(f"Sigma-Itr: {it} | Err. {res:.4e}\n")
        f.write("\n")

# ============================= Grains ====================================
import numpy as np
from scipy.spatial import Voronoi, KDTree

def generate_periodic_voronoi_grains(
        Nx: int, 
        Ny: int, 
        num_grains: int, 
        device: str = "cpu"
    ) -> torch.Tensor:
    """
    Generate a periodic grain structure using Voronoi tessellation.
    
    Args:
        Nx, Ny: Grid dimensions
        num_grains: Number of grains
        device: Device for the output tensor
    
    Returns:
        grain_structure: (Nx, Ny) tensor with grain IDs (periodic)
    """
    np.random.seed(43)
    x = np.random.rand(num_grains) * Nx
    y = np.random.rand(num_grains) * Ny
    base_points = np.column_stack([x, y])
    
    shifts = np.array([
        [0, 0],    # Original
        [Nx, 0],   # Right
        [-Nx, 0],  # Left
        [0, Ny],    # Top
        [0, -Ny],  # Bottom
        [Nx, Ny],  # Top-right
        [Nx, -Ny], # Bottom-right
        [-Nx, Ny], # Top-left
        [-Nx, -Ny] # Bottom-left
    ])

    periodic_points = np.vstack([base_points + shift for shift in shifts])
    grain_ids = np.tile(np.arange(num_grains), len(shifts))  

    tree = KDTree(periodic_points)

    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
    query_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    _, nearest_indices = tree.query(query_points, k=1)
    nearest_grain_ids = grain_ids[nearest_indices]
    
    grain_structure = torch.from_numpy(nearest_grain_ids.reshape(Nx, Ny)).to(device)
    return grain_structure

import matplotlib.pyplot as plt

def plot_hysteresis_cycles(hysteresis_filename, metadata_filename, results_folder, save_path="PE_loops.png"):
    # Construct file paths
    base_dir = os.path.dirname(os.getcwd())
    hysteresis_path = os.path.join(base_dir, results_folder, hysteresis_filename)
    metadata_path = os.path.join(base_dir, results_folder, metadata_filename)

    # Load simulation parameters
    with h5py.File(metadata_path, 'r') as hf:
        sim_params_group = hf['Simulation_Parameters']
        sim_params = {key: sim_params_group[key][()] for key in sim_params_group}

    # Load hysteresis data
    with h5py.File(hysteresis_path, "r") as f:
        E_applied = f["E_applied"][:]  # Electric field
        Py = f["Py"][:]                # Polarization

    # Extract timing and step information
    dt = sim_params['dt']
    nsteps = sim_params['nsteps']
    T_sim = dt * nsteps
    T_cycle = T_sim / 2  # Two full cycles → 1 cycle = T_sim / 2

    nsteps_per_cycle = int(T_cycle / dt)

    # Extract first and second cycle
    E1 = E_applied[:nsteps_per_cycle]
    P1 = Py[:nsteps_per_cycle]
    E2 = E_applied[nsteps_per_cycle:2 * nsteps_per_cycle]
    P2 = Py[nsteps_per_cycle:2 * nsteps_per_cycle]

    # Plot
    plt.figure(figsize=(6, 5))
    plt.plot(E1, P1, label="1st Cycle", color="blue")
    plt.plot(E2, P2, label="2nd Cycle", color="red")

    plt.xlabel("Electric Field (E)")
    plt.ylabel("Polarization (P)")
    plt.title("P-E Loops: First and Second Cycles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

    plt.show()