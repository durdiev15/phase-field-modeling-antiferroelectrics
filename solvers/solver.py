import torch
from torch.fft import fftn as fft, ifftn as ifft
from torch.fft import fftfreq
from math import pi, isnan, isinf
import itertools
import sys

class FourierSolver:
    def __init__(self, 
                 max_iter: int,
                 tol: float,
                 dtype=torch.float32, 
                 device:torch.device=torch.device("cpu")
                 ):
        """
        Initialize the Fourier solver with default FFT data type.
        
        Args:
            dtype_fft: Data type to use for FFT operations (default: torch.complex64)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.dtype = dtype
        self.device = device
        self.dtype_fft = torch.complex64 if self.dtype == torch.float32 else torch.complex128

    def fourier_frequencies(self, 
                          Nx: int, 
                          dx: float, 
                          Ny: int, 
                          dy: float
                          ) -> torch.Tensor:
        """
        Compute Fourier frequencies for a 2D grid.
        
        Args:
            Nx: Number of grid points in x-direction
            dx: Grid spacing in x-direction
            Ny: Number of grid points in y-direction
            dy: Grid spacing in y-direction
            
        Returns:
            freq: Frequency tensor of shape (2, Nx, Ny)
        """
        kx = (2.0 * pi * fftfreq(Nx, dx)).to(self.dtype).to(self.device)
        ky = (2.0 * pi * fftfreq(Ny, dy)).to(self.dtype).to(self.device)

        kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
        freq = torch.stack((kx_grid, ky_grid))

        return freq
    
    def green_operator(self, 
                      C0: torch.Tensor,  # (2, 2, 2, 2)
                      freq: torch.Tensor  # (2, Nx, Ny)
                      ) -> torch.Tensor:
        """
        Compute the Green's operator for elasticity in Fourier space.
        
        Args:
            C0: Stiffness tensor (2,2,2,2)
            freq: Frequency tensor (2, Nx, Ny)
            
        Returns:
            G_elas: Green's operator (2,2,2,2,Nx,Ny)
        """
        A = torch.einsum('pijq, px..., qx... -> ijx...', C0.to(self.dtype), freq, freq)

        adjG = torch.empty_like(A)
        adjG[0, 0] = A[1, 1]
        adjG[1, 1] = A[0, 0]
        adjG[0, 1] = -A[0, 1]
        adjG[1, 0] = -A[1, 0]

        detG = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        detG[0, 0] = 1.0  # Avoid division by zero at zero frequency

        invA = adjG / detG

        G_elas = torch.zeros((2, 2, 2, 2, *freq[0].shape), dtype=self.dtype, device=self.device)
        for i, j, k, l in itertools.product(range(2), repeat=4):
            G_elas[i, j, k, l] = 0.25 * (invA[i, l] * freq[k] * freq[j] + 
                                        invA[j, l] * freq[k] * freq[i] + 
                                        invA[i, k] * freq[l] * freq[j] + 
                                        invA[j, k] * freq[i] * freq[l])
            
        del A, adjG, detG, invA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return G_elas
    
    def solve_mechanics(self, 
                       C0: torch.Tensor,             # (2,2,2,2) stiffness tensor
                       G: torch.Tensor,              # (2,2,2,2,Nx,Ny) Green's operator
                       eps0: torch.Tensor,           # (2,2,Nx,Ny) spontaneous strain
                       eps_ext: torch.Tensor        # (2,2,Nx,Ny) applied external strain (macro strain)
                       ):
        """
        Solve the mechanical equilibrium equation, div sigma = 0, iteratively in Fourier space.
        
        Returns:
            sigma: Stress field (2,2,Nx,Ny)
            eps_elas: Elastic strain field (2,2,Nx,Ny)
            div_sigma: Divergence of stress field (2,Nx,Ny)
        """
        C0_complex = C0.to(dtype=self.dtype_fft)
        G_complex = G.to(dtype=self.dtype_fft)
        eps0_fft = fft(eps0, dim=(2,3))
        
        tau = -torch.einsum('ijkl, klx... -> ijx...', C0_complex, eps0_fft)

        eps_tot_fft = -torch.einsum('ijklx..., klx... -> ijx...', G_complex, tau) 
        eps_tot_fft[:,:,0,0] = 0

        eps_tot = ifft(eps_tot_fft, dim=(2,3)).real

        eps_elas = eps_tot + eps_ext - eps0

        sigma = torch.einsum('ijkl, klx... -> ijx...', C0, eps_elas)

        del eps_tot, eps0_fft, tau, eps_tot_fft, C0_complex, G_complex

        return sigma, eps_elas
    
    def solve_mechanics_pol(self, 
                            Cx: torch.Tensor,             # (2,2,2,2,Nx,Ny) heterogeneous stiffness tensor
                            C0: torch.Tensor,             # (2,2,2,2) homogeneous stiffness tensor
                            G: torch.Tensor,              # (2,2,2,2,Nx,Ny) Green's operator
                            eps0: torch.Tensor,           # (2,2,Nx,Ny) spontaneous strain
                            eps_ext: torch.Tensor,         # (2,2,Nx,Ny) applied external strain (macro strain)
                            number_interations = 100, tol = 1e-4):
        """
        Solve the mechanical equilibrium equation, div sigma = 0, iteratively in Fourier space.
        
        Returns:
            sigma: Stress field (2,2,Nx,Ny)
            eps_elas: Elastic strain field (2,2,Nx,Ny)
            div_sigma: Divergence of stress field (2,Nx,Ny)
        """
        C0_complex = C0.to(dtype=self.dtype_fft)
        G_complex = G.to(dtype=self.dtype_fft)
        eps0_fft = fft(eps0, dim=(2,3))

        sig_norm = 1e-8
        eps_tot = torch.zeros_like(eps0, dtype=eps0.dtype, device=eps0.device)

        for itr in range(number_interations):

            eps_elas = eps_tot + eps_ext - eps0
            sigma = torch.einsum('ijklx..., klx... -> ijx...', Cx, eps_elas)
            sig_norm_new = torch.norm(sigma.ravel(), p=2).item()

            err_s = abs((sig_norm_new - sig_norm) / sig_norm)
            print(f"Iteration: {itr} | S. error = {err_s:.2E}")

            if isnan(err_s) is True or isinf(err_s) is True:
                print(f"Iteration loop terminated due to the presence of NaN or Inf ")
                sys.exit(1)
            elif err_s < tol:
                break

            sig_norm = sig_norm_new

            tau = sigma - torch.einsum('ijkl, klx... -> ijx...', C0, eps_elas)
            tau_fft = fft(tau, dim=(2,3))

            alpha = tau_fft - torch.einsum('ijkl, klx... -> ijx...', C0_complex, eps0_fft)
            eps_tot_fft = -torch.einsum('ijklx..., klx... -> ijx...', G_complex, alpha)

            eps_tot = ifft(eps_tot_fft, dim=(2,3)).real
        
        del eps_tot, eps0_fft, tau
        del tau_fft, alpha, eps_tot_fft, C0_complex, G_complex

        return sigma, eps_elas
    
    def solve_electrostatics(self,
                            K: float,              # permittivity
                            freq: torch.Tensor,    # (2, Nx, Ny) Fourier frequencies
                            E_ext: torch.Tensor,   # (2, Nx, Ny) applied field
                            P_fft: torch.Tensor   # (2, Nx, Ny) polarization in Fourier space
                            ):
        """
        Solve the electrostatic equation, divD = 0, iteratively in Fourier space.
        
        Returns:
            E: Electric field (2, Nx, Ny)
            phi: Electric potential (Nx, Ny)
            div_D: Divergence of electric displacement field (Nx, Ny)
        """
        freq = freq.to(dtype=self.dtype_fft)
        freq2 = torch.einsum('i..., i...', freq, freq)
        freq2[0, 0] = 1

        phi_fft = -1j * torch.einsum('i..., i...', freq, P_fft) / (K * freq2)
        
        E_fft = -1j * freq * phi_fft

        E = ifft(E_fft, dim=(1,2)).real + E_ext

        del freq2, E_fft

        return E, ifft(phi_fft).real

    # def solve_mechanics(self, 
    #                    C0: torch.Tensor,             # (2,2,2,2) stiffness tensor
    #                    G: torch.Tensor,              # (2,2,2,2,Nx,Ny) Green's operator
    #                    eps0: torch.Tensor,           # (2,2,Nx,Ny) spontaneous strain
    #                    xi: torch.Tensor,             # (2,Nx,Ny) Fourier frequencies (kx, ky)
    #                    eps_ext: torch.Tensor        # (2,2,Nx,Ny) applied external strain (macro strain)
    #                    ):
    #     """
    #     Solve the mechanical equilibrium equation, div sigma = 0, iteratively in Fourier space.
        
    #     Returns:
    #         sigma: Stress field (2,2,Nx,Ny)
    #         eps_elas: Elastic strain field (2,2,Nx,Ny)
    #         div_sigma: Divergence of stress field (2,Nx,Ny)
    #     """

    #     tau = -torch.einsum('ijkl,klxy->ijxy', C0, eps0)
    #     tau_hat = fft(tau, dim=(-2, -1))
    #     eps_hat = -torch.einsum('ijklmn,klmn->ijmn', G, tau_hat)

    #     eps_ext_mean = fft(eps_ext, dim=(-2, -1))[:, :, 0, 0]
    #     eps_hat[:, :, 0, 0] = eps_ext_mean # enforce mean strain
    #     eps = ifft(eps_hat, dim=(-2, -1)).real

    #     itr_total, res_data = [], []

    #     alpha = 1.9
    #     prev_res = float('inf')
        
    #     for itr in range(self.max_iter):
    #         eps_elas = eps - eps0
    #         sigma = torch.einsum('ijkl,klxy->ijxy', C0, eps_elas)
    #         sigma_hat = fft(sigma, dim=(-2, -1))

    #         div_sigma_hat = 1j * torch.einsum('ijxy,jxy->ixy', sigma_hat, xi)
    #         div_sigma = ifft(div_sigma_hat, dim=(1, 2)).real

    #         # res = torch.mean(div_sigma_hat**2).sqrt().item() / torch.mean(sigma_hat[:, :, 0, 0]**2).sqrt().item()
    #         res = torch.mean(div_sigma ** 2).sqrt().item()

    #         itr_total.append(itr)
    #         res_data.append(res.real)

    #         if res.real < self.tol:
    #             break

    #         # if res > prev_res:
    #         #     alpha *= 0.8  # Slow down if oscillating
    #         # else:
    #         #     alpha = min(alpha * 1.15, 2.0)  # Careful not to overshoot
    #         # prev_res = res
            
    #         eps_hat -= alpha * torch.einsum('ijklmn,klmn->ijmn', G, sigma_hat)
    #         eps_hat[:, :, 0, 0] = eps_ext_mean
    #         eps = ifft(eps_hat, dim=(2, 3)).real

    #     del eps_hat, sigma_hat, div_sigma_hat, tau, xi, G

    #     return sigma, eps_elas, div_sigma, itr_total, res_data
    
    # def solve_electrostatics(self,
    #                         K: float,              # permittivity
    #                         freq: torch.Tensor,    # (2, Nx, Ny) Fourier frequencies
    #                         E_ext: torch.Tensor,   # (2, Nx, Ny) applied field
    #                         P_fft: torch.Tensor   # (2, Nx, Ny) polarization in Fourier space
    #                         ):
    #     """
    #     Solve the electrostatic equation, divD = 0, iteratively in Fourier space.
        
    #     Returns:
    #         E: Electric field (2, Nx, Ny)
    #         phi: Electric potential (Nx, Ny)
    #         div_D: Divergence of electric displacement field (Nx, Ny)
    #     """
    #     freq2 = torch.einsum('i...,i...->...', freq, freq)
    #     freq2[0, 0] = 1e-8

    #     phi_hat = -1j * torch.einsum('i..., i...->...', freq, P_fft) / (K * freq2)

    #     itr_total, res_data = [], []

    #     alpha = 1.5
    #     prev_res = float('inf')

    #     for it in range(self.max_iter):
    #         E_hat = -1j * freq * phi_hat 
    #         E = ifft(E_hat, dim=(-2, -1)).real + E_ext

    #         D = K * E + ifft(P_fft, dim=(-2, -1)).real
    #         D_hat = fft(D, dim=(-2, -1))
    #         div_D_hat = 1j * torch.einsum('ixy,ixy->xy', D_hat, freq)
    #         div_D = ifft(div_D_hat).real

    #         res = torch.mean(div_D ** 2).sqrt().item()
            
    #         itr_total.append(it)
    #         res_data.append(res)

    #         if res < self.tol:
    #             break

    #         # if res > prev_res:
    #         #     alpha *= 0.8  # Slow down if oscillating
    #         # else:
    #         #     alpha = min(alpha * 1.05, 2.0)  # Careful not to overshoot
    #         # prev_res = res

    #         phi_hat -= alpha * div_D_hat / (K * freq2)

    #     del freq2, div_D_hat

    #     return E, ifft(phi_hat).real, div_D, D, itr_total, res_data
