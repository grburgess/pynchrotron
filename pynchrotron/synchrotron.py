import numba as nb
import numpy as np

from pynchrotron.synchrotron_kernel import (
    synchrotron_kernel,
    compute_synchtron_matrix,
)


@nb.njit(fastmath=True, parallel=False)
def cool_and_radiate(
    energy,
    n_photon_energies,
    ne,
    B,
    bulk_lorentz_factor,
    gamma_min,
    gamma_max,
    index,
    DT,
    n_grid_points,
    steps,
):

    # allocate needed grids
    
    gamma = np.zeros(n_grid_points)
    gamma2 = np.zeros(n_grid_points)
    fgamma = np.zeros(n_grid_points)
    G = np.zeros(n_grid_points + 1)
    source_eval = np.zeros(n_grid_points)
    emission = np.zeros(n_photon_energies)

    # precompute some variables
    
    pm1 = 1 - index
    denom = np.power(gamma_max, pm1) - np.power(gamma_min, pm1)
    cool = 1.29234e-9 * B * B

    # define the step size such that we have a grid slightly
    # out gamma max to avoid boundary issues

    step = np.exp(1.0 / n_grid_points * np.log(gamma_max * 1.1))

    # now compute the grid and the source input
    # which is a power law
    
    for j in range(n_grid_points):

        gamma[j] = np.power(step, j)
        gamma2[j] = gamma[j] * gamma[j]

        if j < n_grid_points - 1:

            G[j] = 0.5 * (gamma[j] + gamma[j] * step)

        else:

            G[n_grid_points - 1] = 0.5 * (gamma[j] + gamma[j] * step)

        if (gamma[j] > gamma_min) and (gamma[j] < gamma_max):
            source_eval[j] = ne * np.power(gamma[j], -index) * (pm1) * 1.0 / denom

    # precompute a matrix for each photon energy and electron grid point
            
    synchrotron_matrix = compute_synchtron_matrix(
        energy, gamma2, B, bulk_lorentz_factor, n_photon_energies, n_grid_points
    )


    # allocate the Chang and Cooper arrays
    
    V3 = np.zeros(n_grid_points)
    V2 = np.zeros(n_grid_points)
    delta_gamma1 = G[n_grid_points - 1] - G[n_grid_points - 2]
    delta_gamma2 = G[1] - G[0]


    # these values will remain constant throughout the cooling
    # so just compute them one
    
    for j in range(n_grid_points - 2, 0, -1):

        delta_gamma = 0.5 * (G[j] - G[j - 1])  # Half steps are at j+.5 and j-.5

        gdotp = cool * gamma2[j + 1]  # Forward  step cooling
        gdotm = cool * gamma2[j]  # Backward step cooling

        V3[j] = (DT * gdotp) / delta_gamma  # Tridiagonal coeff.
        V2[j] = 1.0 + (DT * gdotm) / delta_gamma  # Tridiagonal coeff.

    # now compute the cooling and synchrotron emission
        
    fgammatp1 = np.zeros(n_grid_points)

    for _ in range(0, steps + 1):


        # set the end point
        
        fgammatp1[n_grid_points - 1] = fgamma[n_grid_points - 1] / (
            1.0
            + (DT * cool * gamma[n_grid_points - 1] * gamma[n_grid_points - 1])
            / delta_gamma1
        )

        # back sweep through the grid 
        
        for j in range(n_grid_points - 2, 0, -1):

            fgammatp1[j] = (fgamma[j] + source_eval[j] + V3[j] * fgammatp1[j + 1]) / V2[
                j
            ]
            fgamma[j] = fgammatp1[j]

        # set the end point
            
        fgammatp1[0] = (
            fgamma[0] + (DT * cool * gamma[1] * gamma[1] * fgammatp1[1]) / delta_gamma2
        )
        fgamma[0] = fgammatp1[0]
        fgamma[-1] = fgammatp1[-1]

        # synchrotron emission

        # now loop through the photon energies and compute the synchrotron
        # cooling with a power law integration
        
        for i in range(n_photon_energies):

            # start the integration
            
            summ = 0.0

            for j in nb.prange(1, n_grid_points):

                summ += synchrotron_matrix[i, j] * fgamma[j] * (gamma[j] - gamma[j - 1])

            emission[i] += summ / (2.0 * energy[i])

    return emission
