from pynchrotron import SynchrotronNumerical
import numpy as np

def test_astromodels_model():

    x = SynchrotronNumerical()

    ene = np.logspace(1,5, 100)

    x(ene)
