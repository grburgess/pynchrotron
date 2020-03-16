import numpy as np
import astropy.units as u
import astropy.constants as constants
from astromodels import Function1D, FunctionMeta

from pynchrotron.synchrotron import cool_and_radiate

__author__ = "grburgess"


class SynchrotronNumerical(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Synchrotron emission from cooling electrions
    latex : $  $
    parameters :
        K :
            desc : normalization
            initial value : 1
            min : 0
    
        B :
            desc : energy scaling
            initial value : 1E2
            min : .01
          
  
        index:
            desc : spectral index of electrons
            initial value : 3.5
            min : 2.
            max : 6
      
        
        gamma_min:
            desc : minimum electron lorentz factor
            initial value : 5E5
            min : 1
            fix: yes
    
        gamma_cool : 
                desc: cooling time of electrons
                initial value: 9E7
                min value: 5E2
         
    
    
        gamma_max:
            desc : minimum electron lorentz factor
            initial value : 1E8
            min : 1E6
            fix: yes

        bulk_gamma:
            desc : bulk Lorentz factor
            initial value : 1
            min : 1.
            fix: yes
       
    """

#    __metaclass__ = FunctionMeta

    def _set_units(self, x_unit, y_unit):

        self.K.unit = y_unit / u.gauss

        self.B.unit = u.gauss

        self.gamma_min.unit = u.dimensionless_unscaled
        self.gamma_min.unit = u.dimensionless_unscaled
        self.gamma_max.unit = u.dimensionless_unscaled
        self.index.unit = u.dimensionless_unscaled

    def evaluate(self, x, K, B, index, gamma_min, gamma_cool, gamma_max, bulk_gamma):

        const_factor = 1.29234e-9
        n_grid = 300
        norm_factor = 1e11

        if isinstance(K, u.Quantity):

            flag = True

            B_ = B.value
            gamma_min_ = gamma_min.value
            gamma_max_ = gamma_max.value
            gamma_cool_ = gamma_cool.value
            index_ = index.value
            unit_ = self.y_unit
            K_ = K.value

            try:
                flag = False
                tmp = len(x)

                x_ = x.value

            except:
                flag = True
                x_ = np.array([x.value])

        else:

            flag = False

            K_, B_, gamma_min_, gamma_cool_, gamma_max_, index_, x_ = (
                K,
                B,
                gamma_min,
                gamma_cool,
                gamma_max,
                index,
                x,
            )
            unit_ = 1.0

        # compute the synchrotron cooling time of the highest
        # energy electron
        sync_cool = 1.0 / (B_ * B_ * const_factor)

        ratio = gamma_max_ / gamma_cool_

        # now we want the total time
        # for an electron at gamma_cool to
        # cool and find out the number of
        # steps that will require

        steps = np.int32(np.round(ratio))

        norm = bulk_gamma * B_ * 3.7797251e-22
        erg2keV = 6.242e8
        if steps == 0:
            out = np.zeros_like(x)

        # norm = bulk_gamma * B_ * 3.7797251E-22
        # erg2keV = 6.242E8
        # if steps == 0:
        #     out = np.zeros_like(x)

        else:

            dt = sync_cool / (gamma_max_)

            out = (
                K_
                * erg2keV
                * norm
                * np.array(
                    cool_and_radiate(
                        x_,
                        len(x_),
                        1.0,
                        B_,
                        bulk_gamma,
                        gamma_min_,
                        gamma_max_,
                        index_,
                        dt,
                        n_grid,
                        steps,
                    )
                )
            )

        return out * unit_ * norm_factor * gamma_cool_ ** 1.5 / gamma_max_
