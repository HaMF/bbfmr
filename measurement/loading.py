# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:44:17 2015

Read TDMS files typically generated by Doll rotate for different measurement
scenarios.

@author: hannes.maierflaig
"""
import logging
logging.basicConfig()
l = logging.getLogger('hamf-console')
import numpy as np
from glob import glob 
from nptdms import TdmsFile

def read2DSweep(filepattern, group, xChan, yChan, signalChan,
                xGroup=None, yGroup=None,
                nXPoints=None, nYPoints=None,
                tdms_file=None):
    """
    Read 2D data from TDMS file, guesses number of points if possible. The re-
    turned data can be directly plotted with imshow() of guiqwt or matplotlib.

    The function assumes that x and y sweep values are the same for every scan.
    Therefore only vectors for x and y are returned, while the returned signal
    is an [len(x) x len(y)] array.

    Parameters
    ----------
    filepattern : string
        filename (relative) of tdms file. may include wilcards (see glob)
    group : string
        datagroup name in tdms file
    xChan : string
        x channel name in tdms file
    xGroup : string, optional
        if set, the xdata is read from this group (prepend "Read." if neccessary)
    yChan : string
        y channel name in tdms file
    yGroup : string, optional
        if set, y data is read from this group (prepend "Read." if neccessary)
    signalChan : string
        signal channel name. signal has to come in len(x) blocks of len(y) entries
    nXPoints : integer, optional
        number of expected points in x direction, gets calculated if not specified
    nYPoints : integer, optional
        number of expected points in y direction, gets calculated if not specified
    tdms_file : TdmsFile, optional
        If set, file with filepattern is ignored and data from tdms_file is used

    Returns
    -------
    vX : dim. (nXPoints x 1) numpy array
        x values
    vY : dim. (nYPoints x 1) numpy array
        y values
    aSignal : (nXPoints x nYPoints) numpy array
        signal as matrix
    tdms_file : TdmsFile
        TdmsFile object of read tdms.
    """

    # Load data from TDMS file
    if not tdms_file:
        files = glob(filepattern)
        if len(files) > 1:
            l.warn("Provided file pattern %s did return multiple results."%filepattern)
        if len(files) < 1:
            l.warn("Provided file pattern %s did not match any files."%filepattern)
        l.info("Reading file %s"%files[0])
        tdms_file = TdmsFile(files[0])

    x = tdms_file.channel_data(xGroup or group, xChan)
    y = tdms_file.channel_data(yGroup or group, yChan)
    signal = tdms_file.channel_data(group, signalChan)

    return list(reshape(x, y, signal, nXPoints, nYPoints)) + [tdms_file]


def read2DFMR(filepattern, group = "Read.PNAX",
              field_channel  = "LS475.Field (T)",
              freq_channel   = "Frequency",
              signal_channel = "MLINearS11",
              phase_channel  = "PHASeS11",
              imag_channel   = None,
              logarithmic    = False,
              nFieldPoints = None, nFreqPoints = None,
              tdms_file=None):
    """
    Read 2D quadrature detected FMR data from TDMS file where it's stored as
    magnitured and phase. Return array of complex values.
    (Convenience wraper for read2DSweep, see there for further documentation)

    Parameters
    ----------
    filepattern : string
        filename (relative) of tdms file. may include wilcards (see glob)
    group : string, (optional, default="Read.PNAX")
        datagroup name in tdms file
    field_channel : string, (optional, default="LS475.Field (T)")
        x channel name in tdms file
    freq_channel : string, (optional, default="Frequency")
        y channel name in tdms file
    signal_channel : string, (optional, default="MLINearS11")
        signal channel name. signal has to come in len(x) blocks of len(y) entries
    phase_channel : string, (optional, default="PHASeS11")
        phase channel. if set data is taken to be recorded as lin.Magnitude and phase

    nFieldPoints : integer, optional
        number of expected points in x direction, gets calculated if not specified
    nFrequencyPoints : integer, optional
        number of expected points in y direction, gets calculated if not specified

    Returns
    -------
    vFields : dim. (nXPoints x 1) numpy array
        x values
    vFrequencies : dim. (nYPoints x 1) numpy array
        y values
    aSignal : (nXPoints x nYPoints) complex numpy array
        signal as matrix
    tdms_file : (TdmsFile)
        loaded tdms file object

    Usage example (matplotlib):
    --------------
    >>> vFields, vFrequencies, aSignal, _ = read2DFMR(filename, nFreqPoints = 3001,
                                                  signalChannel="MLINearS11",
                                                  field_channel="LS475.Field (T)")
    >>> B, f = np.meshgrid(vFields, vFrequencies)
    >>> fig = plt.figure(1)
    >>> fig.clf()
    >>> mesh = pcolormesh(B, f/1e9, aSignal.T, cmap="spectral")
    >>> symmetricClim = np.max([np.abs(np.min(aSignal)), np.abs(np.max(aSignal))])
    >>> mesh.set_clim(-symmetricClim, symmetricClim)
    """
    if not phase_channel and not imag_channel:
        return read2DSweep(filepattern, group, field_channel, freq_channel,
                       signal_channel, nXPoints=nFieldPoints, nYPoints=nFreqPoints,
                       tdms_file=tdms_file)

    elif phase_channel and not imag_channel:
        vFields, vFrequencies, magnitude, tdms_file = read2DSweep(filepattern, group, field_channel, freq_channel,
                       signal_channel, nXPoints=nFieldPoints, nYPoints=nFreqPoints,
                       tdms_file=tdms_file)

        _, _, phase, _ = read2DSweep(filepattern, group, field_channel, freq_channel,
                        phase_channel, nXPoints=nFieldPoints, nYPoints=nFreqPoints, tdms_file=tdms_file)
        if "log" in signal_channel.lower(): magnitude = np.power(10,magnitude/20)
        complexSignal = magnitude * np.exp(1j*np.deg2rad(phase))
        return (vFields, vFrequencies, complexSignal, tdms_file)
    elif imag_channel:
        vFields, vFrequencies, real, tdms_file = read2DSweep(filepattern, group, field_channel, freq_channel,
                        signal_channel, nXPoints=nFieldPoints, nYPoints=nFreqPoints,
                        tdms_file=tdms_file)

        _, _, imag, _ = read2DSweep(filepattern, group, field_channel, freq_channel,
                        imag_channel, nXPoints=nFieldPoints, nYPoints=nFreqPoints, tdms_file=tdms_file)
        complexSignal = real + 1j*imag
        return (vFields, vFrequencies, complexSignal, tdms_file)



def readAngularFMR(filepattern, nFieldPoints=None, nAnglePoints=None,
              groupI="Read.FMRI",
              groupQ="Read.FMRQ",
              fieldChannel="LS475.Field (T)",
              angleChannel="angle.Setpoint [deg]",
              signalChannel="K2010U",
              tdms_file=None):
    """
    Read angle dependent fixed-frequency FMR data from TDMS file where it's
    stored as real an imaginary signal
    (Convenience wrapper for read2DSweep(). See there for further documentation)

    Parameters
    ----------
    filepattern : string
        filename (relative) of tdms file. may include wilcards (see glob)
    groupI : string
        datagroup name in tdms file
    groupQ : string
        datagroup name in tdms file
    fieldChannel : string
        x channel name in tdms file
    angleChannel : string
        y channel name in tdms file
    signalChannel : string
        signal channel name. signal has to come in len(x) blocks of len(y) entries
    nFieldPoints : integer, optional
        number of expected points in x direction, gets calculated if not specified
    nAnglesPoints : integer, optional
        number of expected points in y direction, gets calculated if not specified
    tdms_file : nptdms.TdmsFile, optional
        if specified no file is loaded (filepattern is ignored) but tdms_file used


    Returns
    -------
    vFields : dim. (nXPoints x 1) numpy array
        x values
    vAngles : dim. (nYPoints x 1) numpy array
        y values
    aSignal : (nXPoints x nYPoints) numpy array
        signal as matrix. may be imaginary

    Usage example (matplotlib):
    --------------
    vFields, vAngles, aSignal, _ = read2DFMR(filename, nFreqPoints = 3001,
                                                  signalChannel="MLINearS11",
                                                  fieldChannel="LS475.Field (T)")
    B, f = np.meshgrid(vFields, vAngles)
    fig = plt.figure(1)
    fig.clf()
    mesh = pcolormesh(B, f/1e9, aSignal.T, cmap="spectral")
    symmetricClim = np.max([np.abs(np.min(aSignal)), np.abs(np.max(aSignal))])
    mesh.set_clim(-symmetricClim, symmetricClim)
    """

    if groupI:
        vFields, vAngles, I, tdms_file = read2DSweep(filepattern, groupI,
                                                     fieldChannel, angleChannel,
                                                     nFieldPoints, nAnglePoints,
                                                     tdms_file)
    if groupQ:
        vFields, vAngles, Q, tdms_file = read2DSweep(filepattern, groupQ,
                                                     fieldChannel, angleChannel,
                                                     nFieldPoints, nAnglePoints,
                                                     tdms_file)
    if groupI and groupQ:
        signal = I + 1j*Q
    elif groupI:
        signal = I
    elif groupQ:
        signal = Q

    return vFields, vAngles, signal, tdms_file

def read2DFMR2magFields(filepattern, path_field_after, 
                        path_field_before, path_frequency, 
                        path_real_signal, path_imag_signal,
                        field_points=None, freq_points=None,
                        tdms_file=None):
    """
    Read a VNA- or Lock-In measurement with a field-before and a field-after channel.
    As an input this routine requires a field_before- and a field_after-channel and
    the groups respectively. For the frequency-channel and the signal-channels it
    assumes that these are in the same group. 
    
    Parameters
    ----------
    filepattern : string
        filename (relative) of tdms file. may include wilcards (see glob)
    path_* : tuple of strings
        path i.e. [group, channel] to the corresponding data
    nFieldPoints : integer, optional
        number of expected points in x direction, gets calculated if not specified
    nFrequencyPoints : integer, optional
        number of expected points in y direction, gets calculated if not specified

    Returns
    -------
    x : dim. (nXPoints x 1) numpy array
        averaged field values from field_before and field_after
    y : dim. (nYPoints x 1) numpy array
        frequency values
    complexSignal : (nXPoints x nYPoints) complex numpy array
        signal as matrix
    tdms_file : (TdmsFile)
        loaded tdms file object

    Usage example (matplotlib):
    --------------
    >>> x, y, aSignal, _ = read2DFMR2magFields(fname, signal_group, group_field_before, 
                                           group_field_after, field_before_channel, field_after_channel, 
                                           freq_channel, signal_channel, 
                                           imag_channel, tdms_file=tdms_file)

    """
    if tdms_file is None:
        files = glob(filepattern)
        if len(files) > 1:
            l.warn("Provided file pattern %s did return multiple results."%filepattern)
        if len(files) < 1:
            l.warn("Provided file pattern %s did not match any files."%filepattern)
        l.info("Reading file %s"%files[0])
        tdms_file = TdmsFile(files[0])

    x = average_channels(tdms_file, path_field_before, path_field_after, True)
    y = tdms_file.channel_data(*path_frequency)
    signal = tdms_file.channel_data(*path_real_signal) + 1j*tdms_file.channel_data(*path_imag_signal)
    
    return list(reshape(x, y, signal, field_points, freq_points)) + [tdms_file]

def average_channels(tdms_file, path_one, path_two, trim=False):
    """
    Average two channels from tdms_file given by their paths [group channel],
    trim result to the shorter of the two.
    """
    one = tdms_file.channel_data(*path_one)
    two = tdms_file.channel_data(*path_two)
    if len(one)==len(two):
        x = (one+two)/2
    elif len(one)>len(two) and trim:
        x = (one[0:len(two)]+two)/2
    elif len(one)<len(two) and trim:
        x = (two[0:len(one)]+one)/2            
    else: 
        raise ValueError("Channel length do not match.")
    return x

def reshape(x, y, z, x_points=None, y_points=None):
    """
    Guess number of points for a set of three values (x,y,z), describing 
    x: 1D vector describing the "x-axis of z" (x_00, x_10, x_20, x_30,..)
    y: 1D vector, all y-values for each x value stacked 
       (y_00, y_01, ..., y_10, y11..., y_20, y21, y22,...)
    z: 2D array of shape (x_points, y_points
    
    Gracefully handle corner cases.
    
    Output three 2D arrays. X and Y are tiled or a meshgrid.
    """
    # Automatically determine # of points
    if x_points is None and y_points is None:
        x_points = np.size(x)
    elif x_points is None and y_points is not None:
        x_points = np.size(z)/y_points
    if y_points is None:
        y_points= np.size(y)/x_points
    if (np.floor(y_points) != y_points) or (np.floor(x_points) != x_points):
        l.warning("Can't autodetect nFieldPoints and nFreqPoints.\
                   Is this an aborted measurement? If so, specify these\
                   variables manually")
        l.warning(np.size(z))
        l.warning("x_points= %.2f" % x_points)
        l.warning("x_points= %.2f" % y_points)

    l.debug(np.size(z))
    l.debug("x_points= %.2f" % x_points)
    l.debug("x_points= %.2f" % y_points)
    x_points = int(x_points)
    y_points = int(y_points)
    
    if len(np.unique(y)) == y_points:
        l.debug("Y data are the same for all X-points. Creating meshgrid")
        y, x = np.meshgrid(y[0:y_points], x, copy=False)
    else:
        l.debug("Y data are not the same for all X-points. Tiling x-values and" 
                "creating true 2D array for y.")
        x = np.tile(x[0:x_points], (y_points, 1))
        y = np.reshape(y, (x_points, y_points)).T

    return x, y, np.reshape(z[0:x_points*y_points], (x_points, y_points))