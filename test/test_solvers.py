import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solvers.solver import FourierSolver
from solvers.utils import (random_polarization_field, voigt_to_full_tensor_2D, spon_strain_derivative)

import torch
from torch.fft import fftn as fft, ifftn as ifft

if __name__ == '__main__':
    # --- Simulation Parameters ---
    Nx, Ny = 128, 96      # Grid resolution (can be non-square)
    dx, dy = 0.1, 0.1     # Grid spacing in each dimension
    k0 = 1.0              # Vacuum permittivity
    C11 = 246.0  # GPa
    C12 = 147.0  # GPa
    C44 = 125.0  # GPa
    Q11 = 2.46
    Q12 = 1.47
    Q44 = 1.25

    dtype = torch.float64 # Set desired precision: float64 for double, float32 for single

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    solver = FourierSolver(dtype, device)

    freq = solver.fourier_frequencies(Nx, dx, Ny, dy)
    freq2 = torch.einsum('i..., i...', freq, freq)

    C = voigt_to_full_tensor_2D(C11, C12, C44).to(dtype).to(device)
    Gamma = solver.green_operator(C, freq)
    P = random_polarization_field(Nx, Ny, freq2)
    eps0, deps0_dP = spon_strain_derivative(Q11, Q12, Q44, P)
    eps_ext = torch.zeros_like(eps0)

    E_ext = torch.zeros_like(P)
    P_fft = torch.fft.fftn(P, dim=(1, 2) )
    E, phi = solver.solve_electrostatics(k0, freq, E_ext, P_fft)
    sigma, eps_elas = solver.solve_mechanics(C, Gamma, eps0, eps_ext)

    solver.electric_displacement_divergence(k0, E, P, freq)
    solver.stress_divergence(sigma, freq)