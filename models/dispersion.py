# -*- coding: utf-8 -*-
from lmfit import Model
import numpy  as np
pi = np.pi

# %%
def resfreq_kittel_ip(x, gamma=28.0249527e9, M_eff=140e3, H_a=0):
    """
    Model the dispersion of a thin film ferromagnet magnetized in plane
    (considering only shape anisotropy), i.e. the Kittel equation[1,2].

    .. math:: f_{res} = \gamma/(2\\pi) \mu_0 / h \sqrt{(H + H_{a})(H + H_{a} + M_{eff})}

    Params:
    =======
    x : numeric [T]
        externally applied magnetic field mu_0 H in Tesla
    gamma : scalar [Hz/T]
        gyromagnetic ratio/(2pi) in Hz/T
    M_eff : scalar [A/m]
        effective magnetization in A/m. Equal to the saturation
        magnetization if only shape anisotropy is relevant.
    H_a : scalar [A/m]
        Additional in plane anisotropy. (default: 0)

    Returns:
    ========
        Resonance frequency at the given magnetic field(s) [Hz]

    References:
    ===========
        [1] Kittel, C. Introduction to Solid State Physics. 
            (John Wiley & Sons, 1995).
        [2] Kittel, C. On the Theory of Ferromagnetic Resonance 
            Absorption. Phys. Rev. 73, 155–161 (1948).
        [3] Shaw, J. M., et al. Precise determination of the spectroscopic 
            g -factor by use of broadband ferromagnetic resonance spectroscopy.
            J. Appl. Phys. 114, 243906 (2013).
    """
    mu_0 = 4*pi*1e-7
    H = x/mu_0
    return gamma*mu_0*np.sqrt(np.abs((H+H_a)*(H+M_eff+H_a)))

class FreqKittelIPModel(Model):
    __doc__ = resfreq_kittel_ip.__doc__
    def __init__(self, *args, **kwargs):
        super().__init__(func=resfreq_kittel_ip, *args, **kwargs)
        self.set_param_hint('g', expr=gamma_to_g_expr(self))
    
    def make_params(self, *args, **kwargs):
        params = super().make_params(*args, **kwargs)
        params["H_a"].vary = False
        return params


def resfreq_kittel_oop(x, gamma=28.0249527e9, M_eff=140e3):
    """
    Model the dispersion of a thin film ferromagnet magnetized perpendicular
    to its plane (considering only shape anisotropy) as a function of the
    external magnetic field i.e. the Kittel equation[1,2].

    .. math:: f_{res}(H) = \gamma/(2\\pi) (B - \\mu_0 M_{eff})}

    Params:
    =======
    x : numeric [T]
        externally applied magnetic field B = mu_0 H in Tesla
    gamma : scalar [Hz/T]
        gyromagnetic ratio/(2pi) in Hz/T
    M_eff : scalar [A/m]
        effective magnetization in A/m. Equal to the saturation
        magnetization if only shape anisotropy is relevant.

    Returns:
    ========
        Resonance frequency at the given magnetic field(s) [Hz]

    References:
    ===========
        [1] Kittel, C. Introduction to Solid State Physics. 
            (John Wiley & Sons, 1995).
        [2] Kittel, C. On the Theory of Ferromagnetic Resonance 
            Absorption. Phys. Rev. 73, 155–161 (1948).
    """
    mu_0 = 4*pi*1e-7
    H = x/mu_0
    return gamma*mu_0*(H - M_eff)

class FreqKittelOOPModel(Model):
    __doc__ = resfreq_kittel_oop.__doc__
    def __init__(self, *args, **kwargs):
        super().__init__(func=resfreq_kittel_oop, *args, **kwargs)
        self.set_param_hint('g', expr=gamma_to_g_expr(self))

        
def resfield_kittel_oop(x, gamma=28.0249527e9, M_eff=140e3):
    """
    Inverse of resfreq_kittel_oop. Calculate field of resonance for fixed freq.
    
    Params:
    =======
    x : numeric [Hz]
        microwave frequency
    gamma : scalar [Hz/T]
        gyromagnetic ratio/(2pi) in Hz/T
    Meff : scalar [A/m]
        effective magnetization in A/m. Equal to the saturation
        magnetization if only shape anisotropy is relevant.

    Returns:
    ========
        Field of resonance for the given microwave frequency [T]

    References:
    ===========
        [1] Kittel, C. Introduction to Solid State Physics. 
            (John Wiley & Sons, 1995).
        [2] Kittel, C. On the Theory of Ferromagnetic Resonance 
            Absorption. Phys. Rev. 73, 155–161 (1948).
    """
    mu_0 = 4*pi*1e-7
    f = x
    return mu_0*M_eff + f/gamma
    

class FieldKittelOOPModel(Model):
    __doc__ = resfield_kittel_oop.__doc__
    def __init__(self, *args, **kwargs):
        super().__init__(func=resfield_kittel_oop, *args, **kwargs)
        self.set_param_hint('g', expr=gamma_to_g_expr(self))

        
def gamma_to_g_expr(model):
    "convert gyromagnetic ratio to g-factor"
    h = 6.62607004e-34
    mu_b = 9.274009994e-24
    fmt = "{factor:.9e}*{prefix:s}gamma"
    return fmt.format(factor=h/mu_b, prefix=model.prefix)
        