from pynchrotron import synchrotron_kernel

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
