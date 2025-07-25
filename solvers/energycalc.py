import torch
from torch.fft import fft2 as fft, ifft2 as ifft

class EnergyCalculator:
    def __init__(self, sim_params):
        self.sim_params = sim_params

    def elastic_energy_and_derivative(self, sigma, eps_elas, deps0_dP):

        dH_elas_dP = torch.einsum('kl..., klm... -> m...', sigma, -deps0_dP)
        H_elas = 0.5 * torch.einsum('ijx...,ijx...->x...', sigma, eps_elas)

        return H_elas, dH_elas_dP

    def electric_energy_and_derivative(self, E, P):

        H_elec = -0.5 * self.sim_params['K'] * torch.einsum('ix...,ix...->x...', E, E) \
                 - torch.einsum('ix...,ix...->x...', E, P)

        return H_elec, -E

    def landau_energy_and_derivative(self, P):
        P0, P1 = P[0], P[1]

        f_landau = (self.sim_params['a1'] * (P0**2 + P1**2) +
                    self.sim_params['a11'] * (P0**4 + P1**4) +
                    self.sim_params['a111'] * (P0**6 + P1**6) +
                    self.sim_params['a12'] * (P0**2 * P1**2) +
                    self.sim_params['a112'] * (P0**4 * P1**2 + P0**2 * P1**4))

        df_dP0 = (6 * P0**5 * self.sim_params['a111'] +
                  4 * P0**3 * self.sim_params['a11'] +
                  2 * P0 * self.sim_params['a1'] +
                  self.sim_params['a112'] * (4 * P0**3 * P1**2 + 2 * P0 * P1**4) +
                  self.sim_params['a12'] * (2 * P0 * P1**2))

        df_dP1 = (6 * P1**5 * self.sim_params['a111'] +
                  4 * P1**3 * self.sim_params['a11'] +
                  2 * P1 * self.sim_params['a1'] +
                  self.sim_params['a112'] * (2 * P0**4 * P1 + 4 * P1**3 * P0**2) +
                  self.sim_params['a12'] * (2 * P0**2 * P1))

        dF_dP = torch.stack((df_dP0, df_dP1))

        del df_dP0, df_dP1

        return f_landau, dF_dP

    def gradient_energy(self, freq, P_fft):
        kx, ky = freq[0], freq[1]

        P1_fft, P2_fft = P_fft[0], P_fft[1]

        P1x = ifft(1j * kx * P1_fft, dim=(0, 1)).real
        P1y = ifft(1j * ky * P1_fft, dim=(0, 1)).real
        P2x = ifft(1j * kx * P2_fft, dim=(0, 1)).real
        P2y = ifft(1j * ky * P2_fft, dim=(0, 1)).real

        P1xx = ifft(-(kx**2) * P1_fft, dim=(0, 1)).real
        P1yy = ifft(-(ky**2) * P1_fft, dim=(0, 1)).real
        P2xx = ifft(-(kx**2) * P2_fft, dim=(0, 1)).real
        P2yy = ifft(-(ky**2) * P2_fft, dim=(0, 1)).real

        term1 = -self.sim_params['sigma_theta2'] * (P1x**2 + P1y**2 + P2x**2 + P2y**2)
        term2 = self.sim_params['g'] * (P1xx**2 + P1yy**2 + P2xx**2 + P2yy**2)

        f_grad_density = term1 + term2
        del P1x, P1y, P2x, P2y, P1xx, P1yy, P2xx, P2yy, term1, term2

        return f_grad_density
