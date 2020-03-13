[![Build Status](https://travis-ci.org/grburgess/pynchrotron.svg?branch=master)](https://travis-ci.org/grburgess/pynchrotron)
[![codecov](https://codecov.io/gh/grburgess/pynchrotron/branch/master/graph/badge.svg)](https://codecov.io/gh/grburgess/pynchrotron)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3544259.svg)](https://doi.org/10.5281/zenodo.3544259)

# pynchrotron
![alt text](https://raw.githubusercontent.com/grburgess/pynchrotron/master/logo.png)

Implements synchrotron emission from cooling electrons. This is the model used in [Burgess et al (2019)](https://www.nature.com/articles/s41550-019-0911-z?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+natastron%2Frss%2Fcurrent+%28Nature+Astronomy%29&utm_content=Google+Feedfetcher). Please cite if you find this code useful for your research.

* This code gets rid of the need for GSL which was originally relied on for a quick computation of the of the synchrotron kernel which is an integral over a  Bessel function. 
* This code has been ported from GSL and written directly in python as well as accelerated with numba
* An astromodels function is also supplied for direct use in 3ML.
