# -*- coding: utf-8 -*-
"""
Provide a class to represent a broad band FMR measurement recorded with a
vector network analyzer.

@author: hannes.maierflaig, stwe
"""

import logging
import re
from nptdms import TdmsFile
from bbFMR.measurement import Measurement
from bbFMR.measurement.loading import read2DFMR, read2DFMR2magFields
logging.basicConfig()
l = logging.getLogger('measurement.vna')

# support versioning of processing operations; expose version of package
from bbFMR._version import get_versions
__version__ = get_versions()['version']
del(get_versions)
    
class VNAMeasurement(Measurement):
    """
    Represents a broad band FMR measurement recorded with a vector network
    analyzer. An attempt is made to autodetect the appropriate channels in the
    TDMS file. Re/Im, Lin Mag/Phase and Log Mag/Phase are supported. The
    autodetected values can be overwritten with the keyword arguments.

    Params
    ======
     group : str (None);
         Data group of the measurement. The channels of this group are used in
         the autodetect algorithm
    field_channel : str (None)
        Field channel (group.field_channel)
    freq_channel : str (None)
        Frequency channel
    signal_channel : str (None)
        Signal channel (either Real channel or Magnitude channel)
    imag_channel : str (None)
        Im(signal) channel. If not None, signal_channel interpreted as Re(sig)
    phase_channel : str(None)
        Phase(Signal) channel. If not None, signal_channel is Mag(sig)
    """
    def __init__(self, fname=None, tdms_file=None, **kwargs):
        super(VNAMeasurement, self).__init__(**kwargs)
        if fname or tdms_file:
            self.load_raw_data(fname, tdms_file=tdms_file, **kwargs)

    def load_raw_data(self, fname, *args, tdms_file=None, **kwargs):
        """
        Load 2D data from tdms file and take care of casting to complex values
        """
        if tdms_file is None:
            tdms_file = TdmsFile(fname)

        infodict_auto = self.get_channel_names(tdms_file, 
                                               group=kwargs.get("group", None))
        infodict = VNAMeasurement.merge_channel_names(infodict_auto, kwargs)
        x, y, aSignal, _ = read2DFMR(fname,
                                    tdms_file=tdms_file,
                                    **infodict)
        self.set_XYZ(x, y, aSignal)

        self.metadata["fname"] = fname
        self.metadata["load_raw_data_args"] = args
        self.metadata["load_raw_data_kwargs"] = kwargs
        self.metadata["xlabel"] = infodict["field_channel"]
        self.metadata["ylabel"] = infodict["freq_channel"]

    def merge_channel_names(infodict, kwargs):
        if "field_channel" in kwargs and kwargs["field_channel"]:
            infodict['field_channel'] = kwargs["field_channel"]
        if "freq_channel" in kwargs and kwargs["freq_channel"]:
            infodict['freq_channel'] = kwargs["freq_channel"]
        if "signal_channel" in kwargs and kwargs["signal_channel"]:
            infodict['signal_channel'] = kwargs["signal_channel"]
        if "imag_channel" in kwargs and kwargs["imag_channel"]:
            infodict['imag_channel'] = kwargs["imag_channel"]

        return infodict

    def get_channel_names(self, tdms_file, group=None):
        """
        Tries to extract the channel names for magnetic field, frequency and
        measurement signal (magnitude, phase, real part, imag. part) from the
        TDMS file.

        Returns the info dictionary. Raises a ValueError if the
        extraction fails.
        """
        #: Matches Read.PNA and Read.PNAX
        group_re = re.compile(r'(?P<group>Read.PNAX?)', re.IGNORECASE)
        #: List of names for frequency, field and measurement data (case insensitive)
        frequency_names = ['Frequency']
        field_names     = ['Field', 'Output Current']
        signal_names    = ['REAL', 'MLIN', 'MLOG']
        other_names     = ['IMAG', 'PHAS', 'PHAS']

        # find the name of the group in the TDMS file
        for s in tdms_file.groups():
            if group:
                # user chose function by
                break
            m = group_re.match(s)
            if m:
                group = m.group('group')
                break
        else:
            raise ValueError("No Read.PNA(X) group found")

        def match_strings(channels, names):
            for channel in channels:
                for name in names:
                    path = channel.path.split('/')[-1]
                    if name.lower() in path.lower():
                        yield path

        channels = tdms_file.group_channels(group)
        freq_channel   = list(match_strings(channels, frequency_names))[0]
        field_channel  = list(match_strings(channels, field_names))[0]
        signal_channel = list(match_strings(channels, signal_names))[0]
        other_channel  = list(match_strings(channels, other_names))[0]

        if ('REAL' in signal_channel and not 'IMAG' in other_channel) or \
           ('MLIN' in signal_channel and not 'PHAS' in other_channel):
            raise ValueError("Invalid combination of measurement formats: "
                             "{0} and {1}".format(signal_channel,
                                                  other_channel))

        if 'REAL' in signal_channel:
            other_channel_name = 'imag_channel'
        else:
            other_channel_name = 'phase_channel'

        infodict =  { 'group':            group,
                      'field_channel':    field_channel,
                      'freq_channel':     freq_channel,
                      'signal_channel':   signal_channel,
                      other_channel_name: other_channel,
                    }
        for k in infodict.keys():
            infodict[k] = infodict[k].strip("'")
        return infodict


class VNAReferencedMeasurement(VNAMeasurement):
    def __init__(self, fname=None, 
                 group="Read.PNAX",
                 group_bg="Read.PNAX-bg",
                 **kwargs):
        super(VNAReferencedMeasurement, self).__init__(**kwargs)
        if fname:
            self.load_raw_data(fname, 
                               group=group,
                               group_bg=group_bg,
                               **kwargs)

    def load_raw_data(self, fname,
                      tdms_file=None,
                      **kwargs):
        """
        Load 2D data from tdms file and take care of casting to complex values
        """
        if tdms_file is None:
            tdms_file = TdmsFile(fname)
                      
        # Select channels for signal trace and retreive data
        infodict_auto = self.get_channel_names(tdms_file, group=kwargs["group"])
        infodict = VNAMeasurement.merge_channel_names(infodict_auto, kwargs)
        x, y, aSignal, _ = read2DFMR(fname,
                                     tdms_file=tdms_file,
                                     **infodict)
        # Select channels for background trace and retreive data
        infodict_auto = self.get_channel_names(tdms_file, group=kwargs["group_bg"])
        infodict = VNAMeasurement.merge_channel_names(infodict_auto, kwargs)
        x_bg, y_bg, aSignal_bg, _ = read2DFMR(fname,
                                              tdms_file=tdms_file,
                                              **infodict)
        self.set_XYZ(x.T, y.T, (aSignal/aSignal_bg))

        self.metadata["fname"] = fname
        self.metadata["load_raw_data_args"] = []
        self.metadata["load_raw_data_kwargs"] = kwargs
        self.metadata["xlabel"] = infodict["field_channel"]
        self.metadata["ylabel"] = infodict["freq_channel"]

        
class VNASeparateFieldsMeasurement(VNAMeasurement):
    def __init__(self, fname=None, 
                       tdms_file=None,
                       path_field_before=["Read.field_before", "Field (T)"],
                       path_field_after=["Read.field_before", "Field (T)"],
                       path_frequency=["Read.PNAX", "Frequency"],
                       path_real_signal=["Read.PNAX", "REALS21"],
                       path_imag_signal=["Read.PNAX", "IMAGinaryS21"],
                       field_points=None, freq_points=None, **kwargs):
        super(VNASeparateFieldsMeasurement, self).__init__(**kwargs)
        if fname or tdms_file:
            self.load_raw_data(fname=fname,
                               tdms_file=tdms_file,
                               path_field_before=path_field_before,
                               path_field_after=path_field_after,
                               path_frequency=path_frequency,
                               path_real_signal=path_real_signal,
                               path_imag_signal=path_imag_signal,
                               field_points=field_points, 
                               freq_points=freq_points)
            
    def load_raw_data(self, fname,
                      tdms_file=None,
                      **kwargs):
        """
        Load 2D data from tdms file with two magnetic field channels.
        
        This function takes the same keyword arguments as the explicit kwargs
        of __init__().
        """
        if tdms_file is None:
            tdms_file = TdmsFile(fname) 
            
        x, y, aSignal, _ = read2DFMR2magFields(fname,
                                               tdms_file=tdms_file,
                                               **kwargs)
        self.set_XYZ(x.T, y.T, aSignal)

        self.metadata["fname"] = fname
        self.metadata["load_raw_data_args"] = []
        self.metadata["load_raw_data_kwargs"] = kwargs
        self.metadata["xlabel"] =  "/".join(kwargs["path_field_before"])
        self.metadata["ylabel"] =  "/".join(kwargs["path_real_signal"])
