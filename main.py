import torch
import os

from domain_evolution import SingleCrystalDomainEvolution
from solvers.utils import initial_polarization

from math import pi, sqrt

def run_simulation(a_afe_scale, initial_results_path=None, hysteresis=False):
    # Materials and simulation parameters
    P_scale = 0.2  # C/m^2
    l_scale = 1E-9  # m
    H_scale = 1E7  # J/m^3
    E_scale = H_scale / P_scale
    K_scale = P_scale / E_scale

    sim_params = {
        'P_scale': P_scale,
        'l_scale': l_scale,
        'H_scale': H_scale,
        'E_scale': E_scale,
        'K_scale': K_scale,
        'a_afe_scale': a_afe_scale,

        'a1': -5.54e7 / (H_scale / P_scale**2),
        'a11': 5.60e8 / (H_scale / P_scale**4),
        'a111': 1.65e9 / (H_scale / P_scale**6),
        'a12': 2.89e8 / (H_scale / P_scale**4),
        'a112': -8.66e8 / (H_scale / P_scale**6),
        'a123': 3.19e10 / (H_scale / P_scale**6),

        'C11': 15.6e10 / H_scale,
        'C12': 9.6e10 / H_scale,
        'C44': 12.7e10 / H_scale,

        'Q11': 0.048 / P_scale**-2,
        'Q12': -0.015 / P_scale**-2,
        'Q44': 0.047 / P_scale**-2,

        'K': 100 * 8.85e-12 / K_scale,
        'ac': 0.416e-9 / l_scale,

        'sigma_theta2': a_afe_scale * 1.75 * (0.416e-9 / l_scale)**2 * 1e7 / (H_scale / P_scale**2),
        'g': 1.25 * (0.416e-9 / l_scale)**4 * 1e7 / (H_scale / P_scale**2),

        'mob': 300,
        't_scale': P_scale**2 / (H_scale * 30),

        'Nx': 128, 'Ny': 128,
        'dx': 0.2, 'dy': 0.2,

        'nsteps': 10000 if hysteresis else 5000,
        'nt': 1000,
        'dt': 2.0E-1,

        'hysteresis': hysteresis,
        'E_max': 60e6 if hysteresis else 0,  # must be in V/m
        'E_max_idx': 1  # must be int x=0, y=1

    }

    # Determine file names based on simulation phase
    if initial_results_path is None:
        P, _ = initial_polarization(sim_params)
        file_name_h5 = f"results_init_{a_afe_scale}.h5"
    else:
        P, _ = initial_polarization(sim_params, initial_results_path)
        file_name_h5 = f"results_continued_hysteresis_{a_afe_scale}.h5"

    n = 2 * pi / (sqrt(1) * sim_params['ac']) * sqrt(2 * sim_params['g']/sim_params['sigma_theta2'])
    print(f"Number of periodicity: {n:.2f}")

    results_dir_name = f"results_single_{sim_params['Nx']}x{sim_params['dx']}"
    simulation = SingleCrystalDomainEvolution(
        P=P,
        sim_params=sim_params,
        results_dir_name=results_dir_name,
        file_name_h5=file_name_h5,
        max_iter=1,
        rtol=1e-4,
        dtype=torch.float64
    )
    
    simulation.run()

    return os.path.join(os.getcwd(), results_dir_name, file_name_h5)

def main():
    a_afe_scales = [2.2]#, 1.5, 2.2, 3.0]
    initial_files = {}

    # First phase: run without electric field
    print("Running initial simulations without electric field...")
    for scale in (a_afe_scales):
        result_file = run_simulation(scale, hysteresis=False)
        initial_files[scale] = result_file


if __name__ == "__main__":
    main()
