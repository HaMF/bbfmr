# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 14:29:59 2016

@author: hannes.maierflaig
"""
import numpy as np
from lmfit.models import LinearModel, LorentzianModel
from bbFMR.complex_model import ComplexModel
from bbFMR.models.complex import ComplexLinearModel
import sympy as s

import scipy.constants as c
from numpy import pi
mu_0 = 4*pi*1e-7
mu_b = c.physical_constants["Bohr magneton"][0]

# %% Field space models     
class VNAFMR_SimpleFieldModel(ComplexModel):
    @staticmethod
    def chi_ip_yy(B, f=1e10, M_eff=140e3, dB=1e-2, gamma=28.024e9, **kwargs):
        """
        Dynamic yy component of the susceptibility tensor acording to Louis
        Eq. (2.44) for a thin film magnetized in plane divided by M
        FIXME: Document limits of this approximation

        B ; [T]
            Magnetic field strength B
        f : [Hz]
            MW frequency
        dH : [T]
            HWHM linewidth
        M_eff : [A/m]
            Effective magnetization (due to anisotropy)
        gamma : [Hz/T]
            gyromagnetic ratio, gamma = g*mu_b/hbar

        """
        dH = dB/2 / mu_0  # FWHM
        H = B/mu_0
        w = 2*pi*f
        H_res_eval = w/(mu_0*gamma)
        chi= (
                (M_eff*(H - 1j*dH/2))
                 /
                ((H - 1j*dH/2)*(H + M_eff - 1j*dH/2) - H_res_eval**2)
            )
        return chi

    def func(x, f=1e10, B_res=0.1, dB=1e-2, gamma=28.024e9, Z=1, phi=0):
        w = 2*pi*f
        H_res0 = w/(mu_0*gamma)
        H_res = B_res/mu_0
        M_eff = (H_res0**2-H_res**2)/H_res
        chi = VNAFMR_SimpleFieldModel.chi_ip_yy(x, f=f, M_eff=M_eff,
                                                dB=dB, gamma=gamma)
        return Z*np.exp(1j*phi) * chi

    def __init__(self, *args, **kwargs):
        super().__init__(func=VNAFMR_SimpleFieldModel.func, *args, **kwargs)

    def make_params(self, *args, **kwargs):
        p = super().make_params(*args, **kwargs)
        p["Z"].min = 0
        p["phi"].min = 0
        p["phi"].max = +2*np.pi
        p["gamma"].min = 0
        p["gamma"].vary = False
        p["dB"].min = 0
        p["phi"].min = -np.pi
        p["phi"].max = +np.pi

        return p

    def guess(self, data, x=None, **kwargs):
        lm = LinearModel()
        y_slope = lm.eval(x=x,
                          params=lm.guess(np.abs(data), x=x))
        center, hwhm, height = guess_peak(np.abs(np.abs(data)-y_slope), x=x)

        pars = self.make_params(Ms=height, B_res=center, dB=hwhm)
        return pars


# %% Frequency space models
class VNAFMR_EllipsoidSphereModel(ComplexModel):
    def func(x, B=1, f_r=1e10, M_s=140e3, df=30e6,
             gamma=28.0249527e9, Z=1, phi=0):
        f = x
        return (1.0*1j*Z*f*(B*M_s*gamma**2*mu_0/(-1.0*1j*df*f - f**2 + f_r**2) - 1.0*1j*M_s*df*gamma*mu_0/(-1.0*1j*df*f - f**2 + f_r**2))*np.exp(1.0*1j*phi))

    def __init__(self, *args, **kwargs):
        super(VNAFMR_EllipsoidSphereModel, self).__init__(
            VNAFMR_EllipsoidSphereModel.func, *args,
            **kwargs)
        self.name = "VNAFMR_EllipsoidSphereModel"

    def fit(self, *args, **kwargs):
        fit_kws_orig = kwargs.pop("fit_kws", {})
        if "diag" not in fit_kws_orig:
            fit_kws = {"diag": [1e9, 1, 1e9, 1e3, 1e6,
                                1e9, 1, 1, 1, 1, 1]}
            fit_kws.update(fit_kws_orig)
        else:
            fit_kws = fit_kws_orig

        return super(VNAFMR_EllipsoidSphereModel, self).fit(*args,
                                                                 fit_kws=fit_kws,
                                                                 **kwargs)
    def calc_Z(self, y, params):
        M_s = params[self.prefix + 'M_s'].value
        B = params[self.prefix +'B'].value
        gamma = params[self.prefix +'gamma'].value
        df = params[self.prefix +'df'].value
        f_r = params[self.prefix +'f_r'].value

        amp_r = (1.0*np.sqrt(1.0*B**2*M_s**2*gamma**4*mu_0**2/(df**2*f_r**2) + 1.0*M_s**2*gamma**2*mu_0**2/f_r**2)*np.abs(f_r))
        return y/amp_r


    def make_params(self, **kwargs):
        p = super(VNAFMR_EllipsoidSphereModel, self).make_params(**kwargs)
        p[self.prefix + "B"].vary = False
        p[self.prefix + "M_s"].vary = False
        p[self.prefix + "gamma"].vary = False
        p[self.prefix + "f_r"].min = 0
        p[self.prefix + "df"].min = 0
        p[self.prefix + "phi"].min = -np.pi*2
        p[self.prefix + "phi"].max = np.pi*2
        p[self.prefix + "phi"].value = 0
        return p

    def guess(self, data, x=None, **kwargs):
        center, hwhm, height = guess_peak_lorentzian(np.abs(data), x=x)
        pars = self.make_params(M_s=1.4e5, f_r=center, df=hwhm, **kwargs)
        pars[self.prefix + 'df'].set(min=0.0)
        pars[self.prefix + 'B'].value = 1
        pars[self.prefix + 'Z'].value = self.calc_Z(height, pars)
        return pars

    def recommended_roi(self, x, data, arg):
        # Typically the data input still contains a background. Subtract this.
        lin_model = ComplexLinearModel()
        lin_params = lin_model.guess(data, x=x)
        p = self.guess(data-lin_model.eval(x=x, params=lin_params), x=x)
        width = p[self.prefix + "df"].value * float(arg)
        fr = p[self.prefix + "f_r"].value
        min_idx = np.argmin(np.abs(x - (fr - width)))
        max_idx = np.argmin(np.abs(x - (fr + width)))
        
        return np.arange(min_idx, max_idx)


class VNAFMR_EllipsoidDerivativeSphereModel(ComplexModel):
    def func(x, B=1, f_r=1e10, M_s=140e3, df=30e6,
             gamma=28.0249527e9, Z=1, phi=0):
        f = x
        return (-1.0*1j*Z*f*(B*M_s*gamma**2*mu_0*(1.0*1j*df + 2*f)/(-1.0*1j*df*f - f**2 + f_r**2)**2 - 1.0*1j*M_s*df*gamma*mu_0*(1.0*1j*df + 2*f)/(-1.0*1j*df*f - f**2 + f_r**2)**2)*np.exp(1.0*1j*phi))

    def __init__(self, *args, **kwargs):
        super(VNAFMR_EllipsoidDerivativeSphereModel, self).__init__(
            VNAFMR_EllipsoidDerivativeSphereModel.func, *args,
            **kwargs)
        self.name = "VNAFMR_EllipsoidDerivativeSphereModel"

    def fit(self, *args, **kwargs):
        fit_kws_orig = kwargs.pop("fit_kws", {})
        if "diag" not in fit_kws_orig:
            fit_kws = {"diag": [1e9, 1, 1e9, 1e3, 1e6,
                                1e9, 1, 1, 1, 1, 1]}
            fit_kws.update(fit_kws_orig)
        else:
            fit_kws = fit_kws_orig

        return super(VNAFMR_EllipsoidDerivativeSphereModel, self).fit(*args,
                                                                   fit_kws=fit_kws,
                                                                   **kwargs)

    def make_params(self, **kwargs):
        p = super(VNAFMR_EllipsoidDerivativeSphereModel, self).make_params(**kwargs)
        p[self.prefix + "B"].vary = False
        p[self.prefix + "M_s"].vary = False
        p[self.prefix + "gamma"].vary = False
        p[self.prefix + "phi"].vary = False
        p[self.prefix + "phi"].min = -np.pi*2
        p[self.prefix + "phi"].max= np.pi*2
        p[self.prefix + "f_r"].min = 0
        p[self.prefix + "df"].min = 0
        return p

    def calc_Z(self, y, params):
        M_s = params[self.prefix + 'M_s'].value
        B = params[self.prefix +'B'].value
        gamma = params[self.prefix +'gamma'].value
        df = params[self.prefix +'df'].value
        f_r = params[self.prefix +'f_r'].value
        amp_r_f = (1.0*np.sqrt(1.0*B**2*M_s**2*gamma**4*mu_0**2/(df**2*f_r**4) + 4.0*B**2*M_s**2*gamma**4*mu_0**2/(df**4*f_r**2) + 1.0*M_s**2*gamma**2*mu_0**2/f_r**4 + 4.0*M_s**2*gamma**2*mu_0**2/(df**2*f_r**2))*np.abs(f_r))
        return y/amp_r_f
        
    def guess(self, data, x=None, **kwargs):
        center, hwhm, height = guess_peak(np.abs(data), x=x)
        pars = self.make_params(Ms=height, f_r=center, df=hwhm, **kwargs)
        pars[self.prefix + 'df'].set(min=0.0)
        pars[self.prefix + 'B'].value = 1
        pars[self.prefix + 'Z'].value = self.calc_Z(height, pars)
        return pars

    def recommended_roi(self, x, data, arg):
        p = self.guess(data, x=x)
        width = p[self.prefix + "df"].value * float(arg)
        fr = p[self.prefix + "f_r"].value
        min_idx = np.argmin(np.abs(x - (fr - width)))
        max_idx = np.argmin(np.abs(x - (fr + width)))
        
        return np.arange(min_idx, max_idx)

        
class VNAFMR_EllipsoidDifferenceQuotientSphereModel(ComplexModel):
    def func(x, B=1, f_r=1e10, M_s=140e3, df=30e6, mod_f=1e6,
             gamma=28.0249527e9, Z=1, phi=0):
        f = x
        return (Z*(B*M_s*gamma**2*mu_0/(-1.0*1j*df*(f + mod_f) + f_r**2 - (f + mod_f)**2) - B*M_s*gamma**2*mu_0/(-1.0*1j*df*(f - mod_f) + f_r**2 - (f - mod_f)**2) - 1.0*1j*M_s*df*gamma*mu_0/(-1.0*1j*df*(f + mod_f) + f_r**2 - (f + mod_f)**2) + 1.0*1j*M_s*df*gamma*mu_0/(-1.0*1j*df*(f - mod_f) + f_r**2 - (f - mod_f)**2))*np.exp(1.0*1j*phi)/(2*mod_f))

    def __init__(self, *args, **kwargs):
        super(VNAFMR_EllipsoidDifferenceQuotientSphereModel, self).__init__(
            VNAFMR_EllipsoidDifferenceQuotientSphereModel.func, *args,
            **kwargs)
        self.name = "VNAFMR_EllipsoidDerivativeSphereModel"

    def fit(self, *args, **kwargs):
        fit_kws_orig = kwargs.pop("fit_kws", {})
        if "diag" not in fit_kws_orig:
            fit_kws = {"diag": [1e9, 1, 1e9, 1e3, 1e6,
                                1e9, 1, 1, 1, 1, 1]}
            fit_kws.update(fit_kws_orig)
        else:
            fit_kws = fit_kws_orig

        return super(VNAFMR_EllipsoidDifferenceQuotientSphereModel, self).fit(*args,
                                                                   fit_kws=fit_kws,
                                                                   **kwargs)

    def make_params(self, **kwargs):
        p = super(VNAFMR_EllipsoidDifferenceQuotientSphereModel, self).make_params(**kwargs)
        p[self.prefix + "B"].vary = False
        p[self.prefix + "M_s"].vary = False
        p[self.prefix + "gamma"].vary = False
        p[self.prefix + "phi"].vary = False
        p[self.prefix + "phi"].min = -np.pi*2
        p[self.prefix + "phi"].max= np.pi*2
        p[self.prefix + "f_r"].min = 0
        p[self.prefix + "df"].min = 0
        p[self.prefix + "mod_f"].min = 0
        p[self.prefix + "mod_f"].vary = False
        return p

    def calc_Z(self, y, params):
        M_s = params[self.prefix + 'M_s'].value
        B = params[self.prefix +'B'].value
        gamma = params[self.prefix +'gamma'].value
        df = params[self.prefix +'df'].value
        f_r = params[self.prefix +'f_r'].value
        mod_f = params[self.prefix +'f_r'].value
        amp_r_f = np.abs(np.sqrt((4.0*df**6*f_r**4 - 8.0*df**6*f_r**2*mod_f**2 + 4.0*df**6*mod_f**4 + 16.0*df**4*f_r**6 - 8.0*df**4*f_r**2*mod_f**4 + 8.0*df**4*mod_f**6 + 128.0*df**2*f_r**6*mod_f**2 - 32.0*df**2*f_r**4*mod_f**4 + 4.0*df**2*mod_f**8 + 256*f_r**6*mod_f**4 - 128*f_r**4*mod_f**6 + 16*f_r**2*mod_f**8)/(1.0*df**8*f_r**8 - 4.0*df**8*f_r**6*mod_f**2 + 6.0*df**8*f_r**4*mod_f**4 - 4.0*df**8*f_r**2*mod_f**6 + 1.0*df**8*mod_f**8 + 16.0*df**6*f_r**8*mod_f**2 - 44.0*df**6*f_r**6*mod_f**4 + 44.0*df**6*f_r**4*mod_f**6 - 20.0*df**6*f_r**2*mod_f**8 + 4.0*df**6*mod_f**10 + 96.0*df**4*f_r**8*mod_f**4 - 176.0*df**4*f_r**6*mod_f**6 + 134.0*df**4*f_r**4*mod_f**8 - 44.0*df**4*f_r**2*mod_f**10 + 6.0*df**4*mod_f**12 + 256.0*df**2*f_r**8*mod_f**6 - 320.0*df**2*f_r**6*mod_f**8 + 176.0*df**2*f_r**4*mod_f**10 - 44.0*df**2*f_r**2*mod_f**12 + 4.0*df**2*mod_f**14 + 256*f_r**8*mod_f**8 - 256*f_r**6*mod_f**10 + 96*f_r**4*mod_f**12 - 16*f_r**2*mod_f**14 + mod_f**16))*np.sqrt(B**2*gamma**2 + 1.0*df**2)*np.abs(M_s)*np.abs(gamma)*np.abs(mu_0)/2)
        return y/amp_r_f
        
    def guess(self, data, x=None, **kwargs):
        center, hwhm, height = guess_peak(np.abs(data), x=x)
        pars = self.make_params(Ms=height, f_r=center, df=hwhm, **kwargs)
        pars[self.prefix + 'df'].set(min=0.0)
        pars[self.prefix + 'Z'].value = self.calc_Z(height, pars)
        return pars

    def recommended_roi(self, x, data, arg):
        p = self.guess(data, x=x)
        width = p[self.prefix + "df"].value * float(arg)
        fr = p[self.prefix + "f_r"].value
        min_idx = np.argmin(np.abs(x - (fr - width)))
        max_idx = np.argmin(np.abs(x - (fr + width)))
        
        return np.arange(min_idx, max_idx)
        
#%% Helper functions
def guess_peak(data, x=None, **kwargs):
    y = np.squeeze(np.real(data))
    x = np.squeeze(x)
    maxy, miny = np.max(y), np.min(y)
    imaxy = np.argmin(np.abs(y-maxy))
    halfmax_vals = np.where(y > (maxy+miny)/2.0)[0]
    hwhm = (x[halfmax_vals[-1]] - x[halfmax_vals[0]])/2.0
    center = x[imaxy]

    return center, hwhm, maxy

def guess_peak_lorentzian(data, x=None, **kwargs):
    y = np.abs(np.squeeze(np.real(data)))
    x = np.squeeze(x)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    # prepare fitting a lorentzian
    m_lin = LinearModel()
    m_lorentzian = LorentzianModel()
    p_lin = m_lin.guess(y, x=x)
    p_lorentzian = m_lorentzian.guess(y-m_lin.eval(x=x, params=p_lin), x=x)

    m = m_lin + m_lorentzian
    p = p_lin + p_lorentzian

    r = m.fit(y, x=x, params=p)

    return (r.best_values["center"],
            r.best_values["sigma"],
            r.best_values["amplitude"]/(np.pi*r.best_values["sigma"]))