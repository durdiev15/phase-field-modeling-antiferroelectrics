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
        Solve the mechanical equilibrium equation, div sigma = 0, in Fourier space.
        
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
    
    def solve_electrostatics(self,
                            K: float,              # permittivity
                            freq: torch.Tensor,    # (2, Nx, Ny) Fourier frequencies
                            E_ext: torch.Tensor,   # (2, Nx, Ny) applied field
                            P_fft: torch.Tensor   # (2, Nx, Ny) polarization in Fourier space
                            ):
        """
        Solve the electrostatic equation, divD = 0, in Fourier space.
        
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