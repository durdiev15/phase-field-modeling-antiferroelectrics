import torch
from torch.fft import fftn as fft, ifftn as ifft
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict

from solvers.save import SimulationOutputManager
from solvers.solver import FourierSolver
from solvers.energycalc import EnergyCalculator

from solvers.utils import *

class SingleCrystalDomainEvolution:
    def __init__(self,
                 P: torch.Tensor,    # initial polarization
                 sim_params: Dict,                # simulation metadata
                 results_dir_name: str,  # directory name where we save data
                 file_name_h5: str,   # file name for saving data in .h5
                 max_iter: int,     # max number of iterations for solvers
                 rtol: float,       # tolerance for convergence check in solvers
                 dtype = torch.float32 # default is 32 
                 ):
        
        """Validate input parameters."""
        if sim_params.get('hysteresis', False):
            if 'E_max_idx' not in sim_params or 'E_max' not in sim_params:
                raise ValueError("E_max_idx and E_max must be specified for hysteresis calculation")
        
        """Initialize basic simulation parameters."""
        self.sim_params = sim_params
        self.results_dir_name = results_dir_name
        self.file_name_h5 = file_name_h5
        self.max_iter = max_iter
        self.rtol = rtol
        self.dtype = dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.P = P.to(self.dtype).to(self.device)

        """Initialize solvers, output managers, and energy calculators."""
        self.solver = FourierSolver(self.max_iter, self.rtol, self.dtype, self.device)
        self.output = SimulationOutputManager(self.results_dir_name, self.file_name_h5, 
                                            max_value=1.0, n=3,
                                            hysteresis_h5_filename=f"hysteresis_{self.sim_params['a_afe_scale']}.h5")
        self.energy_calc = EnergyCalculator(self.sim_params)
        
        self.eps_ext = torch.zeros((2, 2, self.sim_params['Nx'], self.sim_params['Ny']), device=self.device)
        self.E_ext = torch.zeros_like(self.P)

        self.C = voigt_to_full_tensor_2D(self.sim_params['C11'], self.sim_params['C12'], self.sim_params['C44']).to(self.dtype).to(self.device)
        self.freq = self.solver.fourier_frequencies(Nx=self.sim_params['Nx'], dx=self.sim_params['dx'], Ny=self.sim_params['Ny'], dy=self.sim_params['dy'])
        self.Gamma = self.solver.green_operator(self.C, self.freq)

        """Precompute denominator for time evolution equation."""
        self.denom = 1 - 2 * self.sim_params['dt'] * self.sim_params['sigma_theta2'] * (self.freq[0]**2 + self.freq[1]**2) \
                       + 2 * self.sim_params['g'] * self.sim_params['dt'] * (self.freq[0]**4 + self.freq[1]**4)
        self._initialize_optional_features()
    
    def calculate_stress_strain(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate stress and strain tensors."""
        eps0, deps0_dP = spon_strain_derivative(self.sim_params['Q11'], self.sim_params['Q12'], self.sim_params['Q44'], self.P)
        sigma, eps_elas, *_ = self.solver.solve_mechanics(C0=self.C, G=self.Gamma, eps0=eps0, eps_ext=self.eps_ext)
        del eps0
        return sigma, eps_elas, deps0_dP
    
    def calculate_electric_field(self, P_fft: torch.Tensor) -> torch.Tensor:
        """Calculate electric field from polarization."""
        E, *_ = self.solver.solve_electrostatics(K=self.sim_params['K'], freq=self.freq, E_ext=self.E_ext, P_fft=P_fft)
        return E
    
    def calculate_energies(self, sigma: torch.Tensor, eps_elas: torch.Tensor,
                          deps0_dP: torch.Tensor, P_fft: torch.Tensor,
                          E: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Calculate all energy contributions and derivatives."""
        H_elas, dH_elas_dP = self.energy_calc.elastic_energy_and_derivative(sigma, eps_elas, deps0_dP)
        H_elec, dH_elec_dP = self.energy_calc.electric_energy_and_derivative(E, self.P)
        H_landau, dH_landau_dP = self.energy_calc.landau_energy_and_derivative(self.P)
        H_grad = self.energy_calc.gradient_energy(self.freq, P_fft)
        
        return H_elas, H_elec, H_landau, H_grad, dH_elas_dP, dH_elec_dP, dH_landau_dP
    
    def step(self, step: int) -> Tuple[torch.Tensor, ...]:
        """Execute a single simulation step."""
        if self.sim_params['hysteresis']:
            self._update_applied_field(step)
        
        P_fft = fft(self.P, dim=(1, 2))
        sigma, eps_elas, deps0_dP = self.calculate_stress_strain()
        E = self.calculate_electric_field(P_fft)
        
        energies = self.calculate_energies(sigma, eps_elas, deps0_dP, P_fft, E)

        self._update_polarization(P_fft, energies)
        
        if self.sim_params['hysteresis']:
            self._store_hysteresis_data(step, eps_elas)
        
        if (step + 1) % self.sim_params['nt'] == 0:
            self._save_and_plot(step + 1)

        del sigma, eps_elas, deps0_dP
            
        return energies

    def run(self):
        """Execute the complete simulation with optional features."""
        self._save_initial_state()
        
        progress_desc = "Simulation Progress"

        progress_bar = tqdm(range(self.sim_params['nsteps']), desc=progress_desc, dynamic_ncols=True)
        for step in progress_bar:
            energies = self.step(step)
            self._track_energies(energies)
            
            if torch.isnan(self.P).any():
                progress_bar.close()
                raise RuntimeError(f"NaN encountered at time step {step}")
                
            del energies
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self._finalize_simulation()

    def _update_applied_field(self, step: int):
        """Update the applied electric field for hysteresis calculation."""
        self.E_ext[self.sim_params['E_max_idx']] = self.E_applied_array[step]

    def _update_polarization(self, P_fft: torch.Tensor, energies: Tuple):
        """Update the polarization field."""
        # *_, dH_elas_dP, dH_elec_dP, dH_landau_dP = energies
        *_, dH_elas_dP, dH_elec_dP, dH_landau_dP = energies

        dH_bulk_dP = dH_elas_dP + dH_elec_dP + dH_landau_dP
        self.P = ifft((P_fft - self.sim_params['dt'] * fft(dH_bulk_dP, dim=(1, 2))) / self.denom, dim=(1, 2)).real

    def _finalize_simulation(self):
        """Handle final simulation tasks and data saving."""
        self.output.save_and_plot_energy_evolution(
            np.array(self.energy_data['H_landau']), 
            np.array(self.energy_data['H_grad']),
            np.array(self.energy_data['H_elas']), 
            np.array(self.energy_data['H_elec'])
        )
        
        if self.sim_params['hysteresis']:
            self._save_hysteresis_data()

    def _save_hysteresis_data(self):
        """Pass data to output manager in expected format."""
        # Convert lists to numpy arrays (minimal overhead at end of simulation)
        E_applied = np.array(self.hysteresis_data['E_applied'])
        Px = np.array(self.hysteresis_data['Px'])
        Py = np.array(self.hysteresis_data['Py'])
        eps_xx = np.array(self.hysteresis_data['eps_xx'])
        eps_yy = np.array(self.hysteresis_data['eps_yy'])
        eps_xy = np.array(self.hysteresis_data['eps_xy'])
        
        # Delegate to output manager
        self.output.save_and_plot_hysteresis(
            E_applied=E_applied,
            Px_data=Px,
            Py_data=Py,
            eps_xx_data=eps_xx,
            eps_yy_data=eps_yy,
            eps_xy_data=eps_xy
        )

    def _initialize_optional_features(self):
        """Initialize features based on simulation parameters."""
        if self.sim_params['hysteresis']:
            self._initialize_hysteresis_arrays()
        self._initialize_energy_tracking()

    def _initialize_energy_tracking(self):
        """Initialize energy tracking arrays."""
        self.energy_data = {
            'H_elas': [],
            'H_elec': [],
            'H_landau': [],
            'H_grad': []
        }

    def _track_energies(self, energies: Tuple):
        """Track energy evolution throughout the simulation."""
        H_elas, H_elec, H_landau, H_grad, *_ = energies
        self.energy_data['H_elas'].append(torch.mean(H_elas).item())
        self.energy_data['H_elec'].append(torch.mean(H_elec).item())
        self.energy_data['H_landau'].append(torch.mean(H_landau).item())
        self.energy_data['H_grad'].append(torch.mean(H_grad).item())

    def _initialize_hysteresis_arrays(self):
        """Initialize storage for hysteresis data using lists for minimal overhead."""
        self.hysteresis_data = {
            'Px': [],          # Polarization x-component
            'Py': [],          # Polarization y-component
            'eps_xx': [],      # Strain xx-component
            'eps_yy': [],      # Strain yy-component
            'eps_xy': [],      # Strain xy-component
            'E_applied': [],   # Applied electric field
            'time': []         # Simulation time steps
        }
        
        T_sim = self.sim_params['dt'] * self.sim_params['nsteps']
        self.t_array = torch.linspace(0, T_sim, self.sim_params['nsteps']).tolist()
        self.E_applied_array = triangular_field_vectorized(
            torch.tensor(self.t_array),
            E_max=self.sim_params['E_max']/self.sim_params['E_scale'],
            T_cycle=T_sim/2
        ).tolist()

        self.output.plot_applied_electric_field(time=self.t_array, E=self.E_applied_array)

    def _store_hysteresis_data(self, step: int, eps_elas: torch.Tensor):
        """Store scalar values for hysteresis tracking."""

        self.hysteresis_data['Px'].append(torch.mean(self.P[0]).item())
        self.hysteresis_data['Py'].append(torch.mean(self.P[1]).item())
        self.hysteresis_data['eps_xx'].append(torch.mean(eps_elas[0, 0]).item())
        self.hysteresis_data['eps_yy'].append(torch.mean(eps_elas[1, 1]).item())
        self.hysteresis_data['eps_xy'].append(torch.mean(eps_elas[0, 1]).item())
        self.hysteresis_data['E_applied'].append(self.E_applied_array[step])
        self.hysteresis_data['time'].append(self.t_array[step])
    
    def _save_initial_state(self):
        """Save initial state of the simulation."""
        self.output.save(0, self.P.cpu().numpy(), SimulationParams=self.sim_params)
        self.output.plot(0, self.P.cpu().numpy())

    def _save_and_plot(self, step: int):
        """Handle saving and plotting operations."""
        P_cpu = self.P.detach().cpu().numpy()
        self.output.save(step, P_cpu)
        self.output.plot(step, P_cpu)
        del P_cpu
