# -*- coding: utf-8 -*-
"""
Damping models for linewidth-frequency and linewidth-temperature.
"""

from lmfit import Model
from lmfit.models import update_param_vals, COMMON_DOC
import numpy  as np

# %% Constants and math helpers
import scipy.constants as c
k_B = c.Boltzmann
pi = np.pi
mu_b = c.physical_constants["Bohr magneton"][0]

def sech(x):
    return 1/np.cosh(x)

def coth(x):
    return np.cosh(x)/np.sinh(x)

# %% Various damping mechanisms and models
def gilbert_damping(x, alpha, df0):
    """
    A Linear model f=x -> alpha* x + df0. Returns the frequency domain half
    width half maximum (HWHM).
    """
    f = x
    return alpha * f + df0

class GilbertDampingModel(Model):
    __doc__ = gilbert_damping.__doc__ + COMMON_DOC if gilbert_damping.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(GilbertDampingModel, self).__init__(gilbert_damping, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        sval, oval = 0., 0.
        if x is not None:
            sval, oval = np.polyfit(x, data, 1)
        pars = self.make_params(df0=oval, alpha=sval)
        return update_param_vals(pars, self.prefix, **kwargs)


def slow_relaxer(x, t=1e-11, T=300, C_re=0.001, M_s=140e3, A=1, E_x=1e-22):
    """
    As in Woltersdorf et al. PRL 102, 257602 (2009). Returns the frequency half
    width half maximum (HWHM).

    Params:
    =======
    x : numeric (Hz)
        Frequency f (= w/2pi).
    t : numeric (s)
        Rare-earth (slow relaxer) relaxation time.
    T : numeric (K)
        Temperature.
    C_re : numeric (1)
        Atomic concentration of rare-earth ions
    A : numeric (J/rad)
        Anisotropy of the 5d-4f exchange interaction
    M_s : numeric (A/m)
        (Volume) magnetization.
    E_x : numeric (J)
        5d-4f exchange energy.
    """
    C = 1e-10*(A*C_re) / (6*M_s*k_B*T)

    return slow_relaxer_simple(x, 1e-9*t, T, C, E_x)

def slow_relaxer_simple(x, t=1e-11, T=100, C=0.001, E_x=1e-22):
    """
    As in Woltersdorf et al. PRL 102, 257602 (2009). Returns the frequency half
    width half maximum (HWHM).

    Params:
    =======
    x : numeric (Hz)
        Frequency f (= w/2pi).
    t : numeric (s)
        Rare-earth (slow relaxer) relaxation time.
    T : numeric (K)
        Temperature.
    C : numeric (1/(rad*s))
        Amplitude parameter. C = \frac{A C_{RE}}{6 M_s k_B T} with variables
        as defined in slow_relaxer()
    """
    f = x
    w = 2*pi*f
    F = sech(E_x/(k_B*T))**2
    #alpha = C*F*((t + 1j*w*t**2)/(1+(w*t)**2))
    alpha_re = C*F*(t/(1+(w*t)**2))

    return w * alpha_re

class SlowRelaxerModel(Model):
    __doc__ = slow_relaxer.__doc__ + COMMON_DOC if slow_relaxer.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(SlowRelaxerModel, self).__init__(slow_relaxer, *args, **kwargs)

    def make_params(self, **kwargs):
        p = super(SlowRelaxerModel, self).make_params(**kwargs)
        if "T" in p.keys():
            # T is not the independent variable
            p["T"].vary = False
        p["M_s"].min = 0
        p["C_re"].min = 0
        p["E_x"].min = 0
        p["t"].min = 0
        return p

    def guess(self, data, x=None, **kwargs):
        pass


class SlowRelaxerSimpleModel(Model):
    __doc__ = slow_relaxer.__doc__ + COMMON_DOC if slow_relaxer.__doc__ else ""
    def __init__(self, *args, **kwargs):
        super(SlowRelaxerSimpleModel, self).__init__(slow_relaxer_simple,
                                               *args, **kwargs)

    def make_params(self, **kwargs):
        p = super(SlowRelaxerSimpleModel, self).make_params(**kwargs)
        if "T" in p.keys():
            p["T"].vary = False
        p["C"].min = 0
        p["E_x"].min = 0
        p["t"].min = 0
        # FIXME: That'd be super nice, but if T is the independent variable
        # it's not available in the asteval namespace
        #p._asteval.symtable["orbach_relaxation"] = orbach_relaxation
        #p["t"].expr = "orbach_relaxation(1e-21, T, E_x)"
        return p

    def guess(self, data, x=None, **kwargs):
        pass
def low_field_losses(x, df_low, f1, n):
    """
    Low field losses due to domain wall movement. Returns the frequency domain
    half width half maximum (HWHM).
    df_low = HWHM
    """
    f = x
    return df_low * (1/(f-f1))**n

class LowFieldLossesModel(Model):
    __doc__ = low_field_losses.__doc__ + COMMON_DOC if low_field_losses.__doc__ else ""

    def __init__(self, *args, **kwargs):
        super(LowFieldLossesModel, self).__init__(low_field_losses, *args, **kwargs)

    def guess(self, data, x=None, **kwargs):
        pass


#%% helper functions
def orbach_relaxation(B, T, D):
    """
    Calculate relaxation rate (1/\tau) (Hz) for an Orbach process
    [B. H. Clarke, K. Teweedale, and R. W. Teale, Phys. Rev. 139, 6A (1965)].

    Params:
    =======
    B : numeric (Hz)
        Positive constant
    D : numeric (J)
        Level splitting.
    T : numeric (K)
        Temperature

    """
    return B/(np.exp(D/(k_B*T)) - 1)

def direct_relaxation(t_0, T, D):
    """
    Calculate relaxation rate (1/\tau) (Hz) for an Orbach process
    [B. H. Clarke, K. Teweedale, and R. W. Teale, Phys. Rev. 139, 6A (1965)].

    Params:
    =======
    t_0 : numeric (Hz)
        Relaxation time at T = 0
    D : numeric (J)
        Level splitting.
    T : numeric (K)
        Temperature

    """
    return (1/t_0) * coth(D/(2*k_B*T))
    
    
def galt_relaxation(t_inf, T, e):
    """
    Calculate relaxation rate (1/\tau) (Hz) according to Galt and Spencer
    [Galt and Spencer, "Loss Mechanisms in Spinel Ferrites", 
     Physical Review 127, 5, (1962)]

    Params:
    =======
    t_inf : numeric (Hz)
        Relaxation time at infinite temperature (?)
    D : numeric (J)
        Activation energy for ion replacement (?)
    T : numeric (K)
        Temperature

    """
    return 1/(t_inf*np.exp(e/(k_B*T)))
