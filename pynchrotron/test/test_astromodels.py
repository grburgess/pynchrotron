from pynchrotron import SynchrotronNumerical
import numpy as np

from astromodels import PointSource, Model

import astropy.units as u


def test_astromodels_model_without_units():

    x = SynchrotronNumerical()

    ene = np.logspace(1,5, 100)

    x(ene)

def test_astromodels_model_with_units():

    x = SynchrotronNumerical()

    ene = np.logspace(1,5, 5) * u.keV

    model = Model(PointSource('test',0,0,spectral_shape=x))

    model.get_point_source_fluxes(0,ene)


    model.get_point_source_fluxes(0, 1*u.keV)
