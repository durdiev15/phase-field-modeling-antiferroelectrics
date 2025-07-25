[![GitHub release (latest by date)](https://img.shields.io/github/v/release/durdiev15/phase-field-modeling-antiferroelectrics)](https://github.com/durdiev15/phase-field-modeling-antiferroelectrics/releases)


# Phase-Field Simulation of Antiferroelectric Materials

This repository contains a Python-based implementation of a phase-field model for simulating domain evolution in antiferroelectric (AFE) materials. The code is built using PyTorch to leverage GPU acceleration and is designed for studying the complex electromechanical behavior of AFE single crystals.

The theoretical framework for this model is detailed in the following publication:
* **Liu, Y., & Xu, B. X. (2020). Insight into perovskite antiferroelectric phases: A Landau theory and phase-field study. *Acta Materialia, 196*, 435-446.** [https://doi.org/10.1016/j.actamat.2020.07.016](https://doi.org/10.1016/j.actamat.2020.07.016)

## Features

* **Time-Dependent Ginzburg-Landau (TDGL) Simulation:** Solves the TDGL equation to model the evolution of the polarization vector field.
* **Electromechanical Coupling:** Fully couples electrostatic and elastic fields, accounting for electrostriction.
* **Fourier Spectral Solver:** Utilizes efficient Fourier-based spectral methods to solve the mechanical and electrostatic equilibrium equations under periodic boundary conditions.
* **Hysteresis Analysis:** Capable of simulating and plotting P-E (Polarization vs. Electric Field) and Strain-E hysteresis loops.
* **Energy Analysis:** Tracks and plots the evolution of Landau, gradient, elastic, and electrostatic energy components throughout the simulation.
* **Data Management:** Saves simulation results in HDF5 format and generates visualizations of the domain structures.

## File Structure

The repository is organized as follows:

```
.
├── main.py                 # Main script to configure and run simulations
├── domain_evolution.py     # Core class for managing the simulation steps and evolution
│
└── solvers/
    ├── __init__.py         # Makes 'solvers' a Python package
    ├── solver.py           # Contains the FourierSolver for mechanical and electrostatic PDEs
    ├── energycalc.py       # Calculates the various energy contributions and their derivatives
    ├── save.py             # Manages all file I/O, including saving data and plotting
    └── utils.py            # Utility functions for initialization, tensor math, etc.
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/durdiev15/phase-field-modeling-antiferroelectrics.git](https://github.com/durdiev15/phase-field-modeling-antiferroelectrics.git)
    cd phase-field-modeling-antiferroelectrics
    ```

2.  **Install dependencies:**
    The code relies on several scientific computing libraries. You can install them using pip. A GPU with CUDA is highly recommended for performance.

    ```bash
    pip install torch numpy matplotlib h5py tqdm scipy
    ```

## How to Run a Simulation

The primary entry point for the simulation is `main.py`.

1.  **Configure Parameters:** Open `main.py` and adjust the simulation parameters within the `run_simulation` function. Key parameters include:
    * `a_afe_scale`: A crucial scaling factor for the anti-ferroelectric term in the Landau potential.
    * `Nx`, `Ny`: Grid dimensions for the simulation.
    * `nsteps`: Total number of simulation time steps.
    * `dt`: Time step size.
    * `hysteresis`: Set to `True` to run a hysteresis simulation with an applied electric field, or `False` for relaxation without a field.
    * `E_max`: The maximum amplitude of the applied electric field (in V/m) for hysteresis simulations.

2.  **Execute the script:**
    ```bash
    python main.py
    ```
    The `main()` function is configured to run an initial relaxation simulation first. You can then modify it to use the output of the first run as the input for a second run with an applied electric field to study hysteresis, as shown in the commented-out sections.

## Output

The simulation generates the following outputs in a directory named `results_single_{Nx}x{dx}`:

* **HDF5 Data Files (`.h5`):**
    * `results_init_{a_afe_scale}.h5`: Contains the full simulation data (polarization, strain, etc.) for the initial relaxation run.
    * `hysteresis_{a_afe_scale}.h5`: Stores the averaged polarization and strain values at each step of the hysteresis simulation.
    * `energy_evolution.h5`: Contains the evolution of each energy component over time.
    * All simulation parameters are saved as metadata within the HDF5 files for reproducibility.

* **Image Files (`.png`):**
    * A sub-directory named `images/` is created to store all plots.
    * **Domain Structure Plots:** Quiver plots showing the polarization vector field at specified intervals.
    * **Hysteresis Loops:** P-E and Strain-E plots.
    * **Energy Evolution Plot:** A plot showing the change in Landau, gradient, elastic, and electrostatic energies over time.
    * **Applied Field Plot:** A plot of the applied electric field waveform.

## Code Breakdown

### `main.py`
This script sets up all physical and numerical parameters for the simulation. It initializes the `SingleCrystalDomainEvolution` class and starts the simulation by calling the `.run()` method.

### `domain_evolution.py`
The `SingleCrystalDomainEvolution` class orchestrates the entire simulation.
* Its `__init__` method sets up the initial state, solvers, and pre-computes necessary quantities like the Green's operator.
* The `run()` method contains the main time-stepping loop.
* The `step()` method executes a single time step, which involves:
    1.  Calculating stress and strain using `solver.solve_mechanics`.
    2.  Calculating the electric field using `solver.solve_electrostatics`.
    3.  Calculating all energy contributions and their derivatives w.r.t. polarization using `energycalc`.
    4.  Updating the polarization field by solving the TDGL equation.

### `solvers/solver.py`
The `FourierSolver` class implements the spectral methods for a 2D system.
* `solve_mechanics`: Solves the mechanical equilibrium equation `div(σ) = 0` by transforming the problem to Fourier space, where the solution becomes an algebraic operation involving the Green's operator.
* `solve_electrostatics`: Solves the electrostatic equilibrium equation `div(D) = 0` in Fourier space.

### `solvers/energycalc.py`
The `EnergyCalculator` class provides methods to compute the different components of the total free energy (Landau, gradient, elastic, electrostatic) and the corresponding thermodynamic driving forces (functional derivatives of the energy with respect to polarization).

### `solvers/save.py`
The `SimulationOutputManager` class handles all file I/O. It creates directories, saves raw simulation data to HDF5 files, and generates all plots using Matplotlib.

### `solvers/utils.py`
This module contains various helper functions, including:
* `initial_polarization`: To set up a random initial polarization field or load one from a previous run.
* `triangular_field_vectorized`: To generate the waveform for the applied electric field during hysteresis simulations.
* `spon_strain_derivative`: To compute the spontaneous strain from the polarization field via electrostriction.
* Functions for converting between Voigt and full tensor notations.

## Releases and Versioning

This project uses semantic versioning (e.g., `v1.0.0`). Stable versions are tagged and can be found on the [Releases page](https://github.com/YOUR_USERNAME/YOUR_REPONAME/releases) of the GitHub repository. The `main` branch reflects the latest stable release.

### Changelog

* **v1.0.0** (2025-07-26)
    * Initial release of the single-crystal AFE phase-field simulation code.
    * Includes solvers for electromechanical coupling and TDGL evolution.
    * Added data saving to HDF5 and plotting capabilities.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
