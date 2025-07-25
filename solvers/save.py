import os
import shutil
import h5py 
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import ndimage

class SimulationOutputManager:
    def __init__(self, 
                 results_dir_name: str,  # directory name where we save data
                 file_name_h5: str,   # file name for saving data in .h5
                 max_value: float,    # vmax for imshow plot
                 n: int,                # grid split for quiver plot 
                 log_filename_mech: str = "mechanics_log.txt",
                 log_filename_elec: str = "electrostatics_log.txt",
                 hysteresis_h5_filename: str = "hysteresis.h5",
                 energy_h5_filename: str = "energy_evolution.h5"):
        
        self.max_value = max_value
        self.n = n
        self.results_dir_name = results_dir_name
        self.file_name_h5 = file_name_h5
        self.log_filename_mech = log_filename_mech
        self.log_filename_elec = log_filename_elec

        self.results_dir = os.path.join(os.getcwd(), results_dir_name) # results directory to save all simulation data
        # self.results_dir = os.path.join(os.getcwd(), results_dir_name, "256x02") 
        os.makedirs(self.results_dir, exist_ok=True)

        self.img_dir = os.path.join(self.results_dir, 'images') # directory to save images 

        self.h5_path = os.path.join(self.results_dir, file_name_h5)
        self.log_path_mech = os.path.join(self.results_dir, log_filename_mech)
        self.log_path_elec = os.path.join(self.results_dir, log_filename_elec)
        self.hysteresis_h5_filename = os.path.join(self.results_dir, hysteresis_h5_filename)
        self.energy_h5_filename = os.path.join(self.results_dir, energy_h5_filename)

        if os.path.exists(self.img_dir):
            shutil.rmtree(self.img_dir)
        os.makedirs(self.img_dir)
        
        if os.path.exists(self.h5_path):
            os.remove(self.h5_path)
            print(f"\nRemoved existing .h5 file: {self.h5_path}")

        if os.path.exists(self.log_path_mech):
            os.remove(self.log_path_mech)
            print(f"Removed existing .txt file: {self.log_path_mech}")

        if os.path.exists(self.log_path_elec):
            os.remove(self.log_path_elec)
            print(f"Removed existing .txt file: {self.log_path_elec}")
        
        if os.path.exists(self.hysteresis_h5_filename):
            os.remove(self.hysteresis_h5_filename)
            print(f"Removed existing .h5 file: {self.hysteresis_h5_filename}")

        if os.path.exists(self.energy_h5_filename):
            os.remove(self.energy_h5_filename)
            print(f"Removed existing .h5 file: {self.energy_h5_filename}\n")

    def save(self, step: int, P: np.array, **kwargs):
        """
        Save simulation data to HDF5 file
        
        Args:
            step: Time step number
            P: Polarization vector field (2D or 3D components)
            kwargs: Additional fields to save (ElasticStrain, ElectricField, SimulationParams)
        """
        with h5py.File(self.h5_path, 'a') as hdf:
            time = f'/time_{int(step)}'

            # Save polarization
            hdf.create_dataset(f'Polarization/Px{time}', data=P[0], compression='gzip')
            hdf.create_dataset(f'Polarization/Py{time}', data=P[1], compression='gzip')
            if len(P) == 3:
                hdf.create_dataset(f'Polarization/Pz{time}', data=P[2], compression='gzip')

            for key, value in kwargs.items():
                if key == 'ElasticStrain':
                    hdf.create_dataset(f'Elastic strain/strain_XX{time}', data=value[0, 0], compression='gzip')
                    hdf.create_dataset(f'Elastic strain/strain_XY{time}', data=value[0, 1], compression='gzip')
                    if value.shape[0] == 3:  # 3D case
                        hdf.create_dataset(f'Elastic strain/strain_XZ{time}', data=value[0, 2], compression='gzip')
                        hdf.create_dataset(f'Elastic strain/strain_YY{time}', data=value[1, 1], compression='gzip')
                        hdf.create_dataset(f'Elastic strain/strain_YZ{time}', data=value[1, 2], compression='gzip')
                        hdf.create_dataset(f'Elastic strain/strain_ZZ{time}', data=value[2, 2], compression='gzip')
                    else:  # 2D case
                        hdf.create_dataset(f'Elastic strain/strain_YY{time}', data=value[1, 1], compression='gzip')

                elif key == 'GrainStructure':
                    # shape must be (Nx, Ny)
                    hdf.create_dataset('GrainStructure', data=value, compression='gzip')

                elif key == 'ElectricField':
                    hdf.create_dataset(f'Electric field/Ex{time}', data=value[0], compression='gzip')
                    hdf.create_dataset(f'Electric field/Ey{time}', data=value[1], compression='gzip')
                    if len(value) == 3:
                        hdf.create_dataset(f'Electric field/Ez{time}', data=value[2], compression='gzip')

                elif key == 'SimulationParams':
                    sim_params_group = hdf.require_group('Simulation_Parameters')
                    for param_key, param_value in value.items():
                        if param_key in sim_params_group:
                            del sim_params_group[param_key]
                        sim_params_group.create_dataset(param_key, data=param_value)
    
    def log_mechanics_data(self, 
                        step: int, 
                        itr_total: list, 
                        res_data: list):
        """
        Append solver iteration residuals to a text log file.

        Args:
            step (int): Current simulation step.
            itr_total (int): Number of iterations for the step.
            res_data (list): List of residuals per iteration.
            log_filename (str): Log file name (default = "mechanics_log.txt").
        """
        with open(self.log_path_mech, "a") as f:
            f.write(f"#--------------------- Step: {step} --------------------------\n")
            for it in range(len(itr_total)):
                f.write(f"Sigma: Itr. {it} | Err. {res_data[it]:.4e}\n")
            f.write("\n")

    def log_electrostatics_data(self, 
                        step: int, 
                        itr_total: list, 
                        res_data: list):
        """
        Append solver iteration residuals to a text log file.

        Args:
            step (int): Current simulation step.
            itr_total (int): Number of iterations for the step.
            res_data (list): List of residuals per iteration.
            log_filename (str): Log file name (default = "mechanics_log.txt").
        """
        with open(self.log_path_elec, "a") as f:
            f.write(f"#--------------------- Step: {step} --------------------------\n")
            for it in range(len(itr_total)):
                f.write(f"D: Itr. {it} | Err. {res_data[it]:.4e}\n")
            f.write("\n")

    # def plot(self, step: int, P: np.array, grain_structure: np.array = None):
    #     P1, P2 = P[0], P[1]
    #     Nx, Ny = P1.shape
    #     # P1 = np.rot90(P1)
    #     # P2 = np.rot90(P2)
        
    #     # Calculate magnitude and direction
    #     P_mag = np.sqrt(P1**2 + P2**2)
    #     direction = np.arctan2(P2, P1)  # Angle in radians [-π, π]
        
    #     # Create figure
    #     fig, ax = plt.subplots(figsize=(15, 10))
        
    #     # Normalize direction to [0,1] for colormap
    #     norm_direction = (direction + np.pi) / (2 * np.pi)  # Convert to [0,1]
        
    #     # Create HSV colormap - hue represents direction, value represents magnitude
    #     hsv = np.zeros((Nx, Ny, 3))
    #     hsv[..., 0] = norm_direction  # Hue (direction)
    #     hsv[..., 1] = 1.0  # Full saturation
    #     hsv[..., 2] = np.clip(P_mag/self.max_value, 0, 1)  # Value (magnitude)
        
    #     # Convert HSV to RGB for display
    #     rgb = matplotlib.colors.hsv_to_rgb(hsv)
        
    #     # Display the polarization field
    #     im = ax.imshow(rgb, origin='lower')

    #     x, y = np.meshgrid(np.arange(Ny), np.arange(Nx))
    #     x, y, u, v = x[::self.n,::self.n],y[::self.n,::self.n],P1[::self.n,::self.n],P2[::self.n,::self.n]
        
    #     ax.quiver(x, y, u, v, color='black', 
    #                 alpha=1, 
    #                 scale=10,              # Increased from 0.15
    #                 scale_units='inches',  # Changed from 'xy'
    #                 width=0.003,
    #                 pivot='mid')           # Added
        
    #     # Add grain boundaries if provided
    #     if grain_structure is not None:
    #         grain_structure = np.rot90(grain_structure)
    #         unique_grains = np.unique(grain_structure)
    #         for grain_id in unique_grains:
    #             mask = (grain_structure == grain_id).astype(float)
    #             ax.contour(mask, levels=[0.5], colors='black',
    #                     linewidths=0.6, extent=[0, Ny, 0, Nx])
        
    #     # Add colorbar for direction
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("bottom", size="5%", pad=0.2)  # Adjust pad for spacing

    #     # Create colorbar using the new axes
    #     cb = matplotlib.colorbar.ColorbarBase(cax, 
    #                                     cmap=matplotlib.cm.hsv,
    #                                     norm=matplotlib.colors.Normalize(-np.pi, np.pi),
    #                                     orientation='horizontal')
    #     cb.set_label('Polarization Direction (radians)')
        
    #     # Add magnitude indicator
    #     ax.text(0.02, 0.95, f'Max magnitude: {np.max(P_mag):.2f}',
    #         transform=ax.transAxes, color='white',
    #         bbox=dict(facecolor='black', alpha=0.5))
        
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
        
    #     save_path = os.path.join(self.img_dir, f"P_step_{step}.png")
    #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close(fig)

    def plot(self, step: int, P: np.array, grain_structure: np.array = None):
        P1, P2 = P[0], P[1]
        Nx, Ny = P1.shape
        # plt.rcParams['image.cmap'] = 'viridis'
        x, y = np.meshgrid(np.arange(Ny), np.arange(Nx))
        u = np.rot90(P1) #ndimage.rotate(P1, 90)
        v = np.rot90(P2) #ndimage.rotate(P2, 90)
        fig, ax = plt.subplots(figsize=(15, 10))
        P_mag = np.sqrt(u**2 + v**2)
        im = ax.imshow(P_mag, cmap='viridis', vmin=0, vmax=self.max_value, alpha=0)
        
        x, y, u, v = x[::self.n,::self.n],y[::self.n,::self.n],u[::self.n,::self.n],v[::self.n,::self.n]

        direction = np.arctan2(v, u)
        ax.quiver(x, y, u, v, direction, 
                    alpha=1, 
                    scale=10,              # Increased from 0.15
                    scale_units='inches',  # Changed from 'xy'
                    width=0.003,
                    pivot='mid')           # Added

        # Add grain boundaries
        if grain_structure is not None:
            grain_structure = np.rot90(grain_structure)
            unique_grains = np.unique(grain_structure)
            for grain_id in unique_grains:
                mask = (grain_structure == grain_id).astype(float)
                ax.contour(mask, levels=[0.5], colors='black',
                        linewidths=1.0, extent=[0, Ny, 0, Nx])

        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        save_path = os.path.join(self.img_dir, f"P_step_{step}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        del x, y, u, v

    

    # def plot(self, step: int, P: np.array, grain_structure: np.array = None):
    #     P1, P2 = P[0], P[1]
    #     Nx, Ny = P1.shape
    #     # plt.rcParams['image.cmap'] = 'viridis'
    #     x, y = np.meshgrid(np.arange(Ny), np.arange(Nx))
    #     # u = np.rot90(P1) #ndimage.rotate(P1, 90)
    #     # v = np.rot90(P2) #ndimage.rotate(P2, 90)
    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     P_mag = np.sqrt(P1**2 + P2**2)
    #     # im = ax.imshow(P_mag, cmap='viridis', vmin=0, vmax=self.max_value, alpha=0)
    #     im = ax.imshow(P_mag, cmap='viridis', origin='lower', 
    #               vmin=0, vmax=self.max_value, alpha=0,
    #               extent=[0, Ny, 0, Nx])  # Explicit extent
        
    #     xx, yy = x[::self.n,::self.n], y[::self.n,::self.n]
    #     uu, vv = P1[::self.n,::self.n], P2[::self.n,::self.n]

    #     direction = np.arctan2(vv, uu)
    #     q = ax.quiver(xx, yy, uu, vv, direction, 
    #              alpha=1, scale=20, scale_units='inches',
    #              width=0.003, pivot='mid',
    #              clim=(-np.pi, np.pi))  # Full circle for direction

    #     # Add grain boundaries
    #     if grain_structure is not None:
    #         grain_structure = ndimage.rotate(grain_structure, 90)
    #         unique_grains = np.unique(grain_structure)
    #         for grain_id in unique_grains:
    #             mask = (grain_structure == grain_id).astype(float)
    #             ax.contour(mask, levels=[0.5], colors='black',
    #                     linewidths=0.6, extent=[0, Ny, 0, Nx])

    #     ax.set_xlim(0, Ny)
    #     ax.set_ylim(0, Nx)
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     save_path = os.path.join(self.img_dir, f"P_step_{step}.png")
    #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close(fig)
    #     del xx, yy, uu, vv


    # def plot(self, step: int, P: np.array, grain_structure: np.array = None):
    #     """
    #     Plot polarization field as quiver plot

    #     Args:
    #         step: Time step number
    #         P: Polarization vector field (2D components)
    #         max_value: Max value for imshow plot
    #         n: Downsampling factor for quiver plot
    #     """
    #     P1, P2 = P[0], P[1]
    #     P_mag = np.sqrt(P1**2 + P2**2)

    #     Nx, Ny = P1.shape
    #     plt.rcParams['image.cmap'] = 'viridis'

    #     x, y = np.meshgrid(np.arange(Ny), np.arange(Nx))  # Note the order (Ny, Nx)

    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     im = ax.imshow(P_mag, cmap='gray', origin='lower', vmin=0, vmax=self.max_value, alpha=0, extent=[0, Ny, 0, Nx])

    #     # Downsample for quiver plot
    #     xx, yy = x[::self.n, ::self.n], y[::self.n, ::self.n]
    #     uu, vv = P1[::self.n, ::self.n], P2[::self.n, ::self.n]
    #     direction = np.arctan2(vv, uu)

    #     ax.quiver(xx, yy, uu, vv, direction, alpha=1, scale=0.35, scale_units='xy', width=0.003)

    #     # Add grain boundaries
    #     if grain_structure:
    #         grain_structure = ndimage.rotate(grain_structure, 90)
    #         unique_grains = np.unique(grain_structure)
    #         for grain_id in unique_grains:
    #             mask = (grain_structure == grain_id).astype(float)
    #             ax.contour(mask, levels=[0.5], colors='black',
    #                     linewidths=0.6, extent=[0, Ny, 0, Nx])

    #     # ax.set_xlim(0, Ny)
    #     # ax.set_ylim(0, Nx)
    #     ax.set_aspect('equal')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     save_path = os.path.join(self.img_dir, f"P_step_{step}.png")
    #     fig.savefig(save_path, dpi=300, bbox_inches='tight')
    #     plt.close(fig)

    #     del xx, yy, uu, vv

    def plot_applied_electric_field(self, time: list, E: list):
        plt.figure(figsize=(8, 4))
        plt.plot(time, E, color='royalblue', linewidth=1.5)
        plt.title("Applied Electric Field Over Time")
        plt.xlabel("Time (~)")
        plt.ylabel("E_applied (~)")
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.img_dir, f"elec_field_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_energy_fields(self, step: int, landau_field: np.array, grad_field: np.array, elas_field: np.array, elec_field: np.array):

        fields = [landau_field, grad_field, elas_field, elec_field]
        titles = ["Landau Energy", "Gradient Energy", 
                "Elastic Energy", "Electrostatic Energy"]
        cmaps = ["Blues", "Oranges", "Greens", "Reds"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, (ax, field, title, cmap) in enumerate(zip(axes, fields, titles, cmaps)):
            im = ax.imshow(field, cmap=cmap, origin="lower")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        plt.suptitle(f"Energy Fields at Step {step}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.img_dir, f"energy_fields_step_{step}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_and_plot_energy_evolution(self, H_landau: np.array, H_grad: np.array, H_elas: np.array, H_elec: np.array):

        filepath = os.path.join(self.results_dir_name, self.energy_h5_filename)
        with h5py.File(filepath, 'w') as h5f:
            h5f.create_dataset('H_landau', data=H_landau, compression="gzip")
            h5f.create_dataset('H_gradient', data=H_grad, compression="gzip")
            h5f.create_dataset('H_elastic', data=H_elas, compression="gzip")
            h5f.create_dataset('H_electrostatic', data=H_elec, compression="gzip")
            h5f.create_dataset('H_total', data=H_landau + H_grad + H_elas + H_elec, compression="gzip")

        steps = np.arange(len(H_landau))
        H_total = H_landau + H_grad + H_elas + H_elec

        # Plot all energies together
        plt.figure(figsize=(10, 6))
        plt.plot(steps, H_landau, label="Landau", color="tab:blue")
        plt.plot(steps, H_grad, label="Gradient", color="tab:orange")
        plt.plot(steps, H_elas, label="Elastic", color="tab:green")
        plt.plot(steps, H_elec, label="Electrostatic", color="tab:red")
        plt.plot(steps, H_total, label="Total", color="black", linestyle="--", linewidth=2)
        plt.xlabel("Simulation step")
        plt.ylabel("Energy density")
        plt.title("Free Energy Evolution Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(self.img_dir, "energy_time.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_and_plot_hysteresis(self, E_applied, Px_data, Py_data,
                              eps_xx_data, eps_yy_data, eps_xy_data):

        # === Save to HDF5 ===
        with h5py.File(self.hysteresis_h5_filename, "w") as f:
            f.create_dataset("E_applied", data=E_applied, compression='gzip')
            f.create_dataset("Px", data=Px_data, compression='gzip')
            f.create_dataset("Py", data=Py_data, compression='gzip')
            f.create_dataset("eps_xx", data=eps_xx_data, compression='gzip')
            f.create_dataset("eps_yy", data=eps_yy_data, compression='gzip')
            f.create_dataset("eps_xy", data=eps_xy_data, compression='gzip')

        # === Plot P-E loops ===
        plt.figure(figsize=(6, 5))
        plt.plot(E_applied, Px_data, label=r"P_x")
        plt.plot(E_applied, Py_data, label=r"P_y")
        plt.xlabel("Electric Field (~)")
        plt.ylabel("Polarization (~)")
        plt.title("P-E Hysteresis Loop")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.img_dir, "P_E_loop.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # === Optional: Plot strain loops ===
        plt.figure(figsize=(6, 5))
        plt.plot(E_applied, eps_xx_data, label=r"$\varepsilon_{xx}$ - E")
        plt.plot(E_applied, eps_yy_data, label=r"$\varepsilon_{yy}$ - E")
        plt.plot(E_applied, eps_xy_data, label=r"$\varepsilon_{xy}$ - E")
        plt.xlabel("Electric Field")
        plt.ylabel("Strain")
        plt.title("Strain-E Hysteresis Loop")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(self.img_dir, "eps_E_loop.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
