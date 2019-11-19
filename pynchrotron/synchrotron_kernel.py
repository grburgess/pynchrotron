import numpy as np
import numba as nb


# this implements the synchrotron kernal from GSL
# without relying on GSL and is faster due to numba


M_SQRT2 = np.sqrt(2.0)
M_SQRT3 = np.sqrt(3.0)
M_PI = np.pi
GSL_SQRT_DBL_EPSILON = 1.4901161193847656e-08
GSL_LOG_DBL_MIN = -7.0839641853226408e02

c0 = M_PI / M_SQRT3

c01 = 0.2257913526447274323630976
cond1 = 2 * M_SQRT2 * GSL_SQRT_DBL_EPSILON
cond3 = -8.0 * GSL_LOG_DBL_MIN / 7.0

# cheb polynomial data

synchrotron1_data = np.array(
    [
        30.364682982501076273,
        17.079395277408394574,
        4.560132133545072889,
        0.549281246730419979,
        0.372976075069301172e-01,
        0.161362430201041242e-02,
        0.481916772120371e-04,
        0.10512425288938e-05,
        0.174638504670e-07,
        0.22815486544e-09,
        0.240443082e-11,
        0.2086588e-13,
        0.15167e-15,
    ]
)

synchrotron1_cs_order = 12
synchrotron1_cs_a = -1.0
synchrotron1_cs_b = 1.0

synchrotron1a_data = np.array(
    [
        2.1329305161355000985,
        0.741352864954200240e-01,
        0.86968099909964198e-02,
        0.11703826248775692e-02,
        0.1645105798619192e-03,
        0.240201021420640e-04,
        0.35827756389389e-05,
        0.5447747626984e-06,
        0.838802856196e-07,
        0.13069882684e-07,
        0.2053099071e-08,
        0.325187537e-09,
        0.517914041e-10,
        0.83002988e-11,
        0.13352728e-11,
        0.2159150e-12,
        0.349967e-13,
        0.56994e-14,
        0.9291e-15,
        0.152e-15,
        0.249e-16,
        0.41e-17,
        0.7e-18,
    ]
)

synchrotron1a_cs_order = 22
synchrotron1a_cs_a = -1.0
synchrotron1a_cs_b = 1.0

synchrotron2_data = np.array(
    [
        0.4490721623532660844,
        0.898353677994187218e-01,
        0.81044573772151290e-02,
        0.4261716991089162e-03,
        0.147609631270746e-04,
        0.3628633615300e-06,
        0.66634807498e-08,
        0.949077166e-10,
        0.1079125e-11,
        0.10022e-13,
        0.77e-16,
        0.5e-18,
    ]
)

synchrotron2_cs_order = 11
synchrotron2_cs_a = -1.0
synchrotron2_cs_b = 1.0


@nb.njit(fastmath=True)
def cheb_eval(coeff, order, a, b, x):
    """
    evaluate the cheb poly for the given value of x

    :param coeff: 
    :param order: 
    :param a: 
    :param b: 
    :param x: 
    :returns: 
    :rtype: 

    """
    d = 0.0
    dd = 0.0

    y = (2.0 * x - a - b) / (b - a)
    y2 = 2.0 * y

    for j in range(order, 0, -1):
        temp = d
        d = y2 * d - dd + coeff[j]
        dd = temp

    temp = d
    d = y * d - dd + 0.5 * coeff[0]

    return d


@nb.njit(fastmath=True, parallel=False)
def synchrotron_kernel(x):
    """
    synchrotron kernel

    :param x: 
    :returns: 
    :rtype: 

    """

    if x < cond1:

        z = np.power(x, 1.0 / 3.0)
        cf = 1 - 8.43812762813205e-01 * z * z
        return 2.14952824153447863671 * z * cf

    elif x <= 4.0:

        px = np.power(x, 1.0 / 3.0)
        px11 = np.power(px, 11)
        t = x * x / 8.0 - 1.0
        result_c1 = cheb_eval(
            synchrotron1_data,
            synchrotron1_cs_order,
            synchrotron1_cs_a,
            synchrotron1_cs_b,
            t,
        )

        result_c2 = cheb_eval(
            synchrotron2_data,
            synchrotron2_cs_order,
            synchrotron2_cs_a,
            synchrotron2_cs_b,
            t,
        )

        return px * result_c1 - px11 * result_c2 - c0 * x

    elif x < cond3:

        t = (12.0 - x) / (x + 4.0)

        result_c1 = cheb_eval(
            synchrotron1a_data,
            synchrotron1a_cs_order,
            synchrotron1a_cs_a,
            synchrotron1a_cs_b,
            t,
        )

        return np.sqrt(x) * result_c1 * np.exp(c01 - x)

    else:

        return 0.0


@nb.njit(fastmath=True, parallel=False)
def compute_synchtron_matrix(
    energy, gamma2, B, bulk_lorentz_factor, n_photon_energies, n_grid_points
):
    """
    compute the evaluation of the synchrotron kernel
    for each photon energy and electron bin

    :param energy: 
    :param gamma2: 
    :param B: 
    :param bulk_lorentz_factor: 
    :param n_photon_energies: 
    :param n_grid_points: 
    :returns: 
    :rtype: 

    """

    # allocate the matrix
    
    out_matrix = np.zeros((n_photon_energies, n_grid_points))

    # compute the synchrotron characteristic energy
    
    ec = B * bulk_lorentz_factor * 1.7365145e-11

    for i in range(n_photon_energies):

        # only compute this once

        arg1 = energy[i] / ec

        for j in range(n_grid_points):

            arg2 = arg1 / gamma2[j]

            # fill the matrix
            
            out_matrix[i, j] = synchrotron_kernel(arg2)

    return out_matrix


__all__ = ["synchrotron_kernel", "compute_synchtron_matrix"]
