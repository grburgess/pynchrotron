from pynchrotron import synchrotron_kernel
from pynchrotron.synchrotron_kernel import cheb_eval


import numpy as np


def test_cheb():


    data = np.array([1E-4,1E-2,1.])

    val = cheb_eval(data,2,-1,1,.1)

    np.testing.assert_almost_equal(val, -0.97895)
    



def test_synchrotron_kernel():


    val = 1E-11

    synchrotron_kernel(val)

    
    val = 1E-5

    synchrotron_kernel(val)

    val = 1E-1

    synchrotron_kernel(val)


    val = 1E2

    synchrotron_kernel(val)

    val = 1E10

    synchrotron_kernel(val)
