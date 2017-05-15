# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 15:58:18 2016

@author: hannes.maierflaig
"""
from bbFMR.complex_model import ComplexModel
import numpy as np
from lmfit.models import update_param_vals

class ComplexLinearModel(ComplexModel):
    @staticmethod
    def func(x, slope_re=1, slope_im=1, intercept_re=0, intercept_im=0):
        slope = slope_re + 1j*slope_im
        intercept = intercept_re + 1j*intercept_im

        return (slope*x + intercept)

    def __init__(self, *args, **kwargs):
        super().__init__(func=ComplexLinearModel.func, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        data = data.flatten()
        slope = (data[-1]-data[0])/(x[-1] - x[0])
        intercept = data[0] - slope*x[0]

        pars = self.make_params(intercept_re=float(np.real(intercept)),
                                intercept_im=float(np.imag(intercept)),
                                slope_re=float(np.real(slope)),
                                slope_im=float(np.imag(slope))),
        return update_param_vals(pars, self.prefix, **kwargs)[0]


class ComplexConstantModel(ComplexModel):
    @staticmethod
    def func(x, offset_re=0, offset_im=0):
        return (offset_re + 1j*offset_im)

    def __init__(self, *args, **kwargs):
        super().__init__(func=ComplexConstantModel.func, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        data = data.flatten()
        offset = np.mean(data)

        pars = self.make_params(offset_re=float(np.real(offset)),
                                offset_im=float(np.imag(offset)))
        return update_param_vals(pars, self.prefix, **kwargs)[0]
