# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:58:30 2016

@author: hannes
"""

import numpy as np

# support versioning of processing operations; expose version of package
from bbFMR._version import get_versions
__version__ = get_versions()['version']
del get_versions


def gradient(X=None, Y=None, Z=None,
             axis:{"type": "int",
                   "min":0, "max": 1,
                   "hint": "Axis along which gradient is calculated (d/dx_{axis})"}=0,
             *args, **kwargs):
    """
    Calculates the gradient along the given axis by calling np.gradient()
    All vargs and kwargs are passed to np.gradient()
    """
    G = np.gradient(Z, *args, **kwargs)
    return X, Y, G[axis]


def derivative_divide(X=None, Y=None, Z=None,
                      axis:{"type": "int",
                            "min":0, "max": 1,
                            "hint": "Axis along which the difference quotient is calculated"}=0,
                      modulation_amp:{"type": "int", "min": 0}=1,
                      average:{"type": "bool"}=True):
    """
    Perform numerical derivative along the axis given by the axis argument and
    divide by the non-derivative value.

    As a background correction method this "dd" method helps in the case where
    the background enters as `V_o`:

    .. math:: S(f,H) = (V_o(f) + Z*V_o(f)*\chi(H))/V_i \\cdot \\exp(\\imath\,\\phi)

    as this results in

    .. math:: \mathrm{dd}(S(f,H)) = Z \\cdot \\frac{\\chi(H)+\\chi(H+\\Delta H)}{\\Delta H} + O(Z^2)

    in the limit of small \Delta H which is equivalent to the partial
    derivative

    .. math:: \\frac{\mathrm{d}\\chi}{\mathrm{d}H} \cdot Z

    Most notably the phase φ which is, in spectroscopic measurements,
    given only by the electrical length and usually complicates data analysis,
    drops out and Z is a scalar real quantity.

    Furthermore a smoothing can be implemented by specifying modulation_amp

    Params:
    =======

    X : array_like (NxM)
        Input X-values (independent variable, M-axis)
    Y : array_like (NxM)
        Input Y-values (independent variable, N-axis)
    Z : array_like (NxM)
        Input Z-values (Signal, dependent variable)
    axis : {0,1}, optional (default: 0)
        The axis, along which the derivative is calculated
    modulation_amp : int, optional (default: 1)
        The number of steps over which the central difference is computed and
        averaged if average is True
    average : bool, optional (default: True)
        If set to True and modulation_amp > 1: Perform operation for 
        modulation_amp 0, 1, 2, 3,... and average the resulting values.
    References:
    ===========
    ...
    """
    if axis == 0:
        delta = np.diff(X, axis=axis)
    elif axis == 1:
        Z = Z.T
        delta = np.diff(Y, axis=axis).T
    else:
        raise(ValueError("Only two dimensional datasets are supported"))

    G = np.zeros_like(Z)
    for row in np.arange(modulation_amp, np.shape(Z)[0]-modulation_amp):
        if average:
            zl = np.mean(Z[row-modulation_amp:row, :], axis=0)
            zh = np.mean(Z[row:row+modulation_amp+1, :], axis=0)
        else:
            zl = Z[row-modulation_amp, :]
            zh = Z[row+modulation_amp, :]
        zm = Z[row, :]
        d = np.mean(delta[row-modulation_amp:row+modulation_amp, :], axis=0)
        
        G[row, :] = (zh-zl)/zm/d

    if axis == 1:
        G = G.T

    return X, Y, G


def cumsum(X=None, Y=None, Z=None,
           axis:{"type": "int",
                 "min":0, "max": 1,
                 "hint": "Axis along which summation is performed"}=0,):
    return X, Y, np.cumsum(Z, axis=axis)


def subtract_slice(X=None, Y=None, Z=None,
                   x_idx:{"type": "int"}=None,
                   y_idx:{"type": "int"}=None,
                   x_val:{"type": "float"}=None,
                   y_val:{"type": "float"}=None):
    """
    Subtract a slice at a given x or y index (by setting x_idx resp. y_idx)
    or at closest to a specific x or y value (by setting x_val or y_val)
    """
    X_cut, Y_cut, Z_cut = cut(X=X, Y=Y, Z=Z,
                              x_idx=x_idx, x_val=x_val, 
                              y_idx=y_idx, y_val=y_val)
    return X, Y, Z-Z_cut
    
def divide_slice(X=None, Y=None, Z=None,
                 x_idx:{"type": "int"}=None,
                 y_idx:{"type": "int"}=None,
                 x_val:{"type": "float"}=None,
                 y_val:{"type": "float"}=None):
    """
    Divide a slice at a given x or y index (by setting x_idx resp. y_idx)
    or at closest to a specific x or y value (by setting x_val or y_val)
    """
    X_cut, Y_cut, Z_cut = cut(X=X, Y=Y, Z=Z,
                              x_idx=x_idx, x_val=x_val, 
                              y_idx=y_idx, y_val=y_val)
    return X, Y, Z/Z_cut

def dual_divide_slice(X=None, Y=None, Z=None,
                      y_val:{"type": "float"}=None,
                      x1_idx:{"type": "int"}=None,
                      x1_val:{"type": "float"}=None,
                      x2_idx:{"type": "int"}=None,
                      x2_val:{"type": "float"}=None):
    """
    Divide a slice at a given x or y index (by setting x_idx resp. y_idx)
    or at closest to a specific x or y value (by setting x_val or y_val)
    """
    X1, Y1, Z1 = cut(X=X, Y=Y, Z=Z, x_idx=x1_idx, x_val=x1_val)
    X2, Y2, Z2 = cut(X=X, Y=Y, Z=Z, x_idx=x2_idx, x_val=x2_val)
    y_idx = np.argmin(np.abs(Y1[0, :]-y_val))
    Z_cut = list(Z1.squeeze()[0:y_idx]) + list(Z2.squeeze()[y_idx:])
    
    return X, Y, Z/Z_cut
    
def subtract(X=None, Y=None, Z=None,
             x_val:{"type": "float"}=None,
             y_val:{"type": "float"}=None,
             z_val:{"type": "float"}=None):
    """
    Divides X, Y and/or Z by a scalar. This can be used e.g.
    to divide the frequency axis by 1e9 to get the frequency in GHz.
    """
    if x_val is not None:
        X -= x_val
    if y_val is not None:
        Y -= y_val
    if z_val is not None:
        Z -= z_val
    return X, Y, Z


def divide(X=None, Y=None, Z=None,
           x_val:{"type": "float"}=None,
           y_val:{"type": "float"}=None,
           z_val:{"type": "float"}=None):
    """
    Divides X, Y and/or Z by a scalar. This can be used e.g.
    to divide the frequency axis by 1e9 to get the frequency in GHz.
    """
    if x_val is not None:
        X /= x_val
    if y_val is not None:
        Y /= y_val
    if z_val is not None:
        Z /= z_val
    return X, Y, Z


def multiply(X=None, Y=None, Z=None,
             x_val:{"type": "float"}=None,
             y_val:{"type": "float"}=None,
             z_val:{"type": "float"}=None):
    """
    Multiplies X, Y and/or Z by a scalar
    """
    if x_val is not None:
        X *= x_val
    if y_val is not None:
        Y *= y_val
    if z_val is not None:
        Z *= z_val
    return X, Y, Z

def add(X=None, Y=None, Z=None,
        x_val:{"type": "float"}=None,
        y_val:{"type": "float"}=None,
        z_val:{"type": "float"}=None):
    """
    Multiplies X, Y and/or Z by a scalar
    """
    if x_val is not None:
        X += x_val
    if y_val is not None:
        Y += y_val
    if z_val is not None:
        Z += z_val
    return X, Y, Z

def conjugate(X=None, Y=None, Z=None):
    """
    Complex conjugate the data (z-value)
    """
    return X, Y, np.conjugate(Z)

def limit(X=None, Y=None, Z=None,
          x_slc: {"type": "slice"}=slice(None),
          y_slc: {"type": "slice"}=slice(None)):
    return X[x_slc, y_slc], Y[x_slc, y_slc], Z[x_slc, y_slc]


def cut(X=None, Y=None, Z=None,
        x_idx=None,
        y_idx=None,
        x_val=None,
        y_val=None):
    """
    Extract a slice at a given x or y index (x_idx, y_idx) or x or y value
    (x_val, y_val)
    FIXME: using slices does not collabs the dimension (as wished) but as it's
    written atm, it fails for xy_idx = 0
    """
    if ((x_idx is not None or x_val is not None) and not
       (y_idx is not None or y_val is not None)):
        if x_val is not None:
            x_idx = np.argmin(np.abs(X[:, 0]-x_val))
        slc = [(x_idx,), slice(None)]
        return X[slc], Y[slc], Z[slc]
    elif y_idx is not None or y_val is not None:
        if y_val is not None:
            y_idx = np.argmin(np.abs(Y[0, :]-y_val))
        slc = [slice(None), (y_idx,)]
        return X[slc], Y[slc], Z[slc]
    else:
        raise(ValueError("Specify one of x_idx, x_val, y_idx or y_val"))


def real(X=None, Y=None, Z=None):
    return X, Y, np.real(Z)


def imag(X=None, Y=None, Z=None):
    return X, Y, np.imag(Z)


def mag(X=None, Y=None, Z=None):
    return X, Y, np.abs(Z)


def abs(X=None, Y=None, Z=None):
    return X, Y, np.abs(Z)


def phase(X=None, Y=None, Z=None):
    return X, Y, np.rad2deg(np.angle(Z))


def limit_rois(X=None, Y=None, Z=None,
               rois=[],
               axis:{"type": "int",
                     "min":0, "max": 1}=0,):
    newshape = list(np.shape(Z))
    if axis == 0:
        newshape[1] = np.abs(rois[0][1] - rois[0][0])
    elif axis == 1:
        newshape[0] = np.abs(rois[0][1] - rois[0][0])
    else:
        raise ValueError("Axis must be 0 or 1")

    X_roi = np.zeros(newshape, dtype=X.dtype)
    Y_roi = np.zeros(newshape, dtype=Y.dtype)
    Z_roi = np.zeros(newshape, dtype=Z.dtype)
    for i, roi in enumerate(rois):
        roi = np.sort(roi)
        if axis == 0:
            X_roi[i, :] = X[i, roi[0]:roi[1]]
            Y_roi[i, :] = Y[i, roi[0]:roi[1]]
            Z_roi[i, :] = Z[i, roi[0]:roi[1]]
        else:
            X_roi[:, i] = X[roi[0]:roi[1], i]
            Y_roi[:, i] = Y[roi[0]:roi[1], i]
            Z_roi[:, i] = Z[roi[0]:roi[1], i]

    return X_roi, Y_roi, Z_roi

def swap_axes(X=None, Y=None, Z=None,
              m=None):
    if m is not None:
        m.metadata["xlabel"], m.metadata["ylabel"] = m.metadata["ylabel"], m.metadata["xlabel"]
    return Y.T, X.T, Z.T

def scaleaxes(X=None, Y=None, Z=None,
                 x_factor:{"type": "float"}=None,
                 y_factor:{"type": "float"}=None,
                 x_offset:{"type": "float"}=None,
                 y_offset:{"type": "float"}=None):
    """
    Axes can be scaled and given an offset (for correction)
    """
    
    if x_factor is not None:
        X *= x_factor
    if x_offset is not None:
        X += x_offset
    if y_factor is not None:
        Y *= y_factor
    if y_offset is not None:
        Y += y_offset 
    return X, Y, Z
    
   
def revert_axis(X=None, Y=None, Z=None):
    return X[::-1,:], Y[::-1,:], Z[::-1,:]

def linear_moving_limit(X=None, Y=None, Z=None,
                        slope:{"type": "float"}=1, 
                        intercept:{"type": "float"}=0, 
                        span:{"type": "int"}=2):
    """
    Slice the data so that only a band that is centered around the linear trend 
    
      .. math:: y_center = slope*x + intercept
    
    remains. The width of the band is span (unitless number of x-points). The 
    width stays the same for all x-values, when x_center approaches the minmum
    or maximum of the x-data (to within span) x_center therefore stays 
    constant.
    
    Params:
    ======
    
    slope : float
        Slope of the linear trend in units of [unit(Y)/unit(X)]
    intercept : float
        Intercept of the linear trend in units of [Y]
    span : int
        Span (number of points)
    """
    if span > Y.shape[1]:
        return X, Y, Z
        
    newshape = (X.shape[0], span-span%2)
    calc_X = np.zeros(newshape, dtype=X.dtype)
    calc_Y = np.zeros(newshape, dtype=Y.dtype)
    calc_Z = np.zeros(newshape, dtype=Z.dtype)
    for i, x in enumerate(X[:, 0]):
        y_center = slope*x+intercept
        y_center_idx = np.argmin(np.abs(Y[i, :]-y_center))
        
        y_min_idx = y_center_idx - int(span/2)
        y_max_idx = y_center_idx + int(span/2)
        if y_min_idx <= 0:
            y_min_idx = 0
            y_max_idx = span
        elif y_max_idx >= Y.shape[1]:
            y_min_idx = Y.shape[1] - span
            y_max_idx = Y.shape[1]
        else:
            y_min_idx = max(y_center_idx - int(span/2), 0)
            y_max_idx = min(y_center_idx + int(span/2), Y.shape[1])         
            
        calc_X[i,:] = X[i, y_min_idx:y_max_idx]
        calc_Y[i,:] = Y[i, y_min_idx:y_max_idx]
        calc_Z[i,:] = Z[i, y_min_idx:y_max_idx]

    return np.array(calc_X), np.array(calc_Y), np.array(calc_Z)

def average_updown(X=None, Y=None, Z=None,
                   n_sweeps:{"type": "int"}=None,
                   antisymmetrize:{"type": "bool"}=False,
                   axis:{"type": "int",
                         "min":0, "max": 1,
                         "hint": "Axis along which gradient is calculated (d/dx_{axis})"}=0):
    """
    Averages consecutive up/down sweeps.
    n_sweeps=1 means one up and one down sweep. If set to None, it is
    automatically detected using X values.
    axis not yet fully implemented
    """
    if antisymmetrize:
        sign_down = -1
    else:
        sign_down = 1

    if n_sweeps is None:
        n_segments = (np.sum((np.diff(np.sign(np.diff(X[:,0]))) != 0)*1) + 2)/2
        n_sweeps = n_segments/2

    n_points = np.shape(Z)[axis]/n_sweeps/2
    if int(n_points) != n_points:
        raise ValueError("Can't divide data into %d sweeps of length %.2f"%(n_sweeps, n_points))

    X_avg = np.zeros((n_points, np.shape(Z)[not axis]))
    Y_avg = np.zeros((n_points, np.shape(Z)[not axis]))
    Z_avg = np.zeros((n_points, np.shape(Z)[not axis]))
    for i in np.arange(n_sweeps):
        start_up = i*2*n_points
        stop_up = (i*2+1)*n_points
        stop_down = stop_up-1
        start_down = (i+1)*2*n_points-1
        slc_up = slice(start_up, stop_up)
        slc_down = slice(start_down, stop_down, -1)

        X_avg += X[slc_up, :]
        X_avg += sign_down*X[slc_down, :]
        Y_avg += Y[slc_up, :]
        Y_avg += sign_down*Y[slc_down, :]
        Z_avg += Z[slc_up, :]
        Z_avg += sign_down*Z[slc_down, :]

    return X_avg/(n_sweeps*2), Y_avg/(n_sweeps*2), Z_avg/(n_sweeps*2)

def referenced_fmr_bare(X=None, Y=None, Z=None,
                   axis:{"type": "int",
                         "min":0, "max": 1,
                         "hint": "Select field axis here (default: 0)"}=0,):
    """
    Expects input data to have alternating rows (axis 0) of
    [data, background, data, background, ...] and calculates
    data/background. Which eliminates magnitude and phase back-
    ground for broadband FMR
    """
    return X[0::2], Y[0::2], Z[0::2]/Z[1::2]

def referenced_fmr(X=None, Y=None, Z=None,
                   delta_x_idx:{"type": "int",
                            "min":0, "max": 1,
                            "hint": "Distance of the background signal (in x-index units)"}=0,):
    """
    For each X-index, calculate Z[X]-X([+delta_x_idx], X will be set to X[X]
    """
    slc_x_val = slice(0, X.shape[0] - delta_x_idx)
    slc_x_delta = slice(delta_x_idx, X.shape[0])
    return X[slc_x_val, :], Y[slc_x_val, :], Z[slc_x_val,:]/Z[slc_x_delta,:]

