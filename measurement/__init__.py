# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 19:19:42 2016

Base class for 1D and 2D measurements

@author: hannes.maierflaig, stwe
"""
from importlib import import_module
import jsonpickle
from datetime import datetime
import os
import inspect
import bbFMR.processing as bp
import numpy as np
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    pass

# support versioning of processing operations; expose version of package
from bbFMR._version import get_versions
__version__ = get_versions()['version']
del(get_versions)

class Measurement(object):
    """
    Prepares data structure, plotting and exporting of spectroscopic
    1D and 2D measurements.
    """

    def __init__(self, X=None, Y=None, Z=None, **kwargs):
        """
        """
        self.raw_X = []
        self.raw_Y = []
        self.raw_Z = []

        self.metadata = {'xlabel': "x",
                         'ylabel': "y",
                         'zlabel': "z",
                         'fname': None,
                         'title': None,
                         "comment": None,
                         "datetime": None
                         }

        if X is not None and Z is not None:
            self.set_XYZ(X, Y, Z)

        self.operations = []
        self.metadata["load_raw_data_args"] = ()
        self.metadata["load_raw_data_kwargs"] = {}

    def set_XYZ(self, X, Y, Z):
        """
        Parse X, Y and Z into correct shapes. X,Y can be a (N,1), (M,1) or
        (N,M) array. Where (N,M is the shape of the data)
        """
        if np.shape(X) == np.shape(Z) and np.shape(Y) == np.shape(Z):
            self.raw_X = X
            self.raw_Y = Y
        elif np.shape(X) == np.shape(Z.T) and np.shape(Y) == np.shape(Z.T):
            self.raw_X = X.T
            self.raw_Y = Y.T
        elif (len(np.shape(X)) == 1 and len(np.shape(Y)) == 1 and Y is not None
              and np.shape(X) != 0 and np.shape(Y) != 0):
            self.raw_Y, self.raw_X = np.meshgrid(Y, X, copy=False)
            if np.shape(self.raw_X) != np.shape(Z):
                raise(ValueError("Could not cast input arrays to\
                                  the data dimensions"))
        elif (len(np.shape(X)) == 1 and len(np.shape(Z)) == 1 and
              len(X) == len(Z)):
            self.raw_Y, self.raw_X = np.meshgrid(1, X, copy=False)
            Z = np.reshape(Z, np.shape(self.raw_X))
        elif (len(np.shape(Y)) == 1 and len(np.shape(Z)) == 1 and
              len(Y) == len(Z)):
            self.raw_X, self.raw_Y = np.meshgrid(Y, 1, copy=False)
            Z = np.reshape(Z, np.shape(self.raw_Y))
        else:
            raise ValueError("Data type not understood. Present 2D data X, Y, "
                             "Z or 1D data in X, Z")

        self.raw_Z = Z
        self.X = self.raw_X
        self.Y = self.raw_Y
        self.Z = self.raw_Z

    def set_xlabel(self, label):
        self.metadata["xlabel"] = label

    def set_ylabel(self, label):
        self.metadata["ylabel"] = label

    def set_zlabel(self, label):
        self.metadata["zlabel"] = label

    def add_operation(self, func, *args, position=None, delay=False, **kwargs):
        """
        Add a function to the processing queue. The data is processed
        immediately if delay is False. Otherwise data is not processed until
        process() is executed

        Params
        ======

        func : callable
            Functions must either accept named parameters X, Y, Z as named
            parameters and also return X,Y,Z or accept Z as first positional
            agument and only return Z.

        position : integer
            Position at which to insert the operation in the processing stack.
            0 is the first operation to be executed, -1 the last.

        delay : integer (default: False)
            Wether to perform the operation right away or to just put it in the
            processing chain and wait for a call to Measuremnt.process()

        *args : tuple
            positional arguments passed to func

        **kwargs : dict
            named arguments passed to func

        FIXME: Catch exceptions here (no __version__)
        """
        if type(func) == np.ufunc:
            # numpy.ufuncs do not return __module__ as implementations vary
            # with the input data. As we're concerned with a "fixed" sequence
            # of operations and thus input data to the function will not change
            # we don't care
            module = "numpy"
        else:
            module = func.__module__

        if position is None:
            position = len(self.operations)

        self.operations.insert(position,
                               {'function': func,
                                'args': args,
                                'kwargs': kwargs,
                                'module': "%s.%s" % (str(module), func.__name__),
                                'version': import_module(module).__version__,
                                })
        if not delay:
            if position is None or position == -1:
                self.run_operation(func, *args, **kwargs)
            else:
                self.process()

    def remove_operation(self, position):
        """
        Remove an operation from the processing queue

        Params
        ======

        position : integer
            Index of the opration in the processing chain that will be deleted.

        """
        del(self.operations[position])

    def find_operation(self, func):
        """
        Return the indices at which a certain operation are carried out.

        Params

        ======

        func : callable
            Operation to look for

        Returns:
        =======

        idx : list
            Indices at which the operation func has been found. If the
            operation has not been found idx is an empty tuple []

        """
        idx = []
        for i, o in enumerate(self.operations):
            if o["function"] == func:
                idx.append(i)
        return idx

    def replace_operation(self, func_search, func_replace,
                          *args, maxreplace=1, **kwargs):
        """
        Replace the operation denoted by func_search by func_replace and the
        new *args and **kwargs for this operation
        """
        idx = self.find_operation(func_search)
        for i in idx:
            self.remove_operation(i)
            self.add_operation(func_replace, position=i, *args, **kwargs)

    def process(self):
        self.X = np.array(self.raw_X)
        self.Y = np.array(self.raw_Y)
        self.Z = np.array(self.raw_Z)

        if self.operations == [] or len(self.raw_X) == 0:
            return

        for operation in self.operations:
            if "args" not in operation:
                operation["args"] = []
            if "kwargs" not in operation:
                operation["kwargs"] = {}
            self.X, self.Y, self.Z = self.run_operation(operation['function'],
                                                        *operation['args'],
                                                        **operation['kwargs'])

    def run_operation(self, func, *args, **kwargs):
        """
        Performs the given operation (without altering X, Y, Z). Returns the
        result of the operation. Parameters are the same as for add_operation.
        """
        argnames = inspect.getfullargspec(func)[0]
        if 'X' in argnames and 'Y' in argnames:
            X, Y, Z = func(X=self.X, Y=self.Y, Z=self.Z,
                           *args, **kwargs)
        else:
            X, Y = self.X, self.Y
            Z = func(self.Z, *args, **kwargs)

        return X, Y, Z

    def plot(self, ax=None, x_step=1, y_step=1, cbar_kwargs={}, *args, **kwargs):
        """
        Plot data. If the data is 1D, a line plot is created. If the data is 2D
        a pcolormesh is created. *args and **kwargs are passed to the
        respective functions (matplotlib.pyplot.plot and
        matplotlib.pyplot.pcolormesh).

        The labels of the axis are chosen according to the metadata of the
        measurement object.

        Params:
        =======
        ax: matplotlib.axis.Axis
            plot to this axis. If None, plot to plt.gca()
        x_step: int
            reduce plotted data by only displaying every x_step-th value
        y_step: int
            reduce plotted data by only displaying every y_step-th value

        Returns:
        ========
        line, None :
            If measurement is 1D: line object
        mesh, cbar :
            If measurement is 2D: mesh and associated colorbar cbar
        """
        if ax is None:
            plt.gcf() # create figure if necessary
            ax = plt.gca()
        if np.shape(self.X)[0] == 1:  # cut at constant x
            line = ax.plot(np.squeeze(self.Y), np.squeeze(self.Z),
                           *args, **kwargs)
            ax.set_xlabel(self.metadata["ylabel"])
            ax.set_ylabel(self.metadata["zlabel"])
            return line, None
        elif np.shape(self.X)[1] == 1:  # cut at constant y
            line = ax.plot(np.squeeze(self.X), np.squeeze(self.Z),
                           *args, **kwargs)
            ax.set_xlabel(self.metadata["xlabel"])
            ax.set_ylabel(self.metadata["zlabel"])
            return line, None
        else:
            mesh = ax.pcolormesh(self.X[0::x_step, 0::y_step],
                                 self.Y[0::x_step, 0::y_step],
                                 self.Z[0::x_step, 0::y_step],
                                 *args, **kwargs)
            ax.set_xlabel(self.metadata["xlabel"])
            ax.set_ylabel(self.metadata["ylabel"])
            cbar = plt.colorbar(mesh, **cbar_kwargs)
            cbar.set_label(self.metadata["zlabel"])

            # Display x, y AND z-value under cursor in matplotlib window
            def format_coord(x, y):
                x_idx = np.argmin(np.abs(self.X[:,0] - x))
                y_idx = np.argmin(np.abs(self.Y[0,:] - y))
                z = self.Z[x_idx, y_idx]

                return "x=%1.4f, y=%1.4f, z=%1.5f" % (x, y, z)
            ax.format_coord = format_coord

            return mesh, cbar

    def save(self, filename=None, data=False,
             squeeze=False, overwrite=False):
        """
        Save processed data and metadata to restore state to several files.
        Also save current library version, as this might be crucial for repro-
        ducible post-processing results as algorithms might change.
        """
        postfix = ".measurement.json"
        if filename is None:
            filename = self.metadata["title"] + postfix
        if not filename.endswith(postfix):
            filename += postfix
        path, basename = os.path.split(os.path.abspath(filename))

        if os.path.isfile(filename):
            if not overwrite:
                raise(FileExistsError)

        with open(filename, "w") as f:
            mdata = self.metadata
            mdata["data_exported"] = "{:%Y-%m-%d %H:%M:%S}".format(datetime.now())
            mdata["version"] = __version__
            f.write(jsonpickle.dumps(self))

        if data:
            self.save_data(os.path.join(path, basename.replace(postfix, "")),
                           squeeze=squeeze, overwrite=overwrite)

    def save_data(self, filename, squeeze=False, overwrite=False):
        """
        Params
        ======
        path : string
            Folder to write data files to
        squeeze : bool
            Write only one file if the data is 1D. In this case path is taken
            as a file name. If the data is a 2D (NxM) array but X and Y are
            given as (1xN) and (1xM) arrays do not save the full (NxM)
            X and Y grids.
            FIXME: Currently not implemented.
        """

        os.makedirs(filename, exist_ok=overwrite)
        basepath, fname = os.path.split(filename)
        path = os.path.join(basepath, fname, "")
        x_fname = path + "/{}.{}.dat".format(fname, self.metadata["xlabel"])
        y_fname = path + "/{}.{}.dat".format(fname, self.metadata["ylabel"])
        z_fname = path + "/{}.{}.dat".format(fname, self.metadata["zlabel"])
        if (os.path.isfile(x_fname) or
            os.path.isfile(y_fname) or
            os.path.isfile(z_fname)):
                if not overwrite:
                    raise(FileExistsError)

        np.savetxt(x_fname, self.X)
        np.savetxt(y_fname, self.Y)
        if self.Z.dtype == complex:
            np.savetxt(z_fname, self.Z.view(float))
        else:
            np.savetxt(z_fname, self.Z)

    def operations_load(self, operations):
        for i, o in enumerate(operations):
            package, fun = o["module"].rsplit('.', 1)
            module = import_module(package)
            operations[i]["function"] = getattr(module, fun)
            operations[i]["args"] = tuple(o["args"])
        self.operations = operations

    def load_raw_data(self, fname, *args, **kwargs):
        """
        Use this method in the implementation of your derived model to load
        the raw data. It will be executed upon unpickeling the object.
        
        Add all arguments needed needed to load the data (i.e. all arguments
        that are passed to this function) to the metadata as demonstrated 
        below.
        """
        self.metadata["fname"] = fname
        self.metadata["load_raw_data_args"] = args
        self.metadata["load_raw_data_kwargs"] = kwargs
        
        raise NotImplementedError()

    def reload(self):
        """
        Reload the raw data from file as has been done when calling
        Measurment.load_raw_data
        """
        self.load_raw_data(self.metadata["fname"],
                           *self.metadata["load_raw_data_args"],
                           **self.metadata["load_raw_data_kwargs"])

    def __setstate__(self, state):
        """
        Restore state of a measurement.

        Params:
        =======
        state :  dict
            see __getstate__() for information on contents of state dict
        """
        self.operations_load(state["operations"])
        self.metadata = state["metadata"]
        self.load_raw_data(self.metadata["fname"],
                           *self.metadata["load_raw_data_args"],
                           **self.metadata["load_raw_data_kwargs"])
        self.process()

    def __getstate__(self):
        """
        Returns
        ======
        state : dict
            The state of the object as dict containing:
            +  "metadata": attribute (dict) of the measurement containing the
               filename of the raw data and arguments and keyword arguments
               required for loading the data, the version of the Measurement
               class, the title of the measurement, x,y and z-label
            +  "operations": chain of the measurement in order to create the
                processed data
        """
        state = {}
        state["metadata"] = self.metadata

        operations = []
        for o in self.operations:
            # strip functions as they can't be json encoded
            operations.append({k: v for k, v in o.items() if k != "function"})
        state["operations"] = operations

        return state

    def load_data(self, path):
        fname = os.path.basename(path)
        x_fname = path + "/{}.{}.dat".format(fname, self.metadata["xlabel"])
        y_fname = path + "/{}.{}.dat".format(fname, self.metadata["ylabel"])
        z_fname = path + "/{}.{}.dat".format(fname, self.metadata["zlabel"])
        self.X = np.loadtxt(x_fname)
        self.Y = np.loadtxt(y_fname)
        self.Z = np.loadtxt(z_fname)
        if np.shape(self.Z) != np.shape(self.X):
            self.Z = self.Z.view(complex)


    def export_ftf(self, fname_stub, axis=0, overwrite=False):
        """
        Export measurement for processing in (labview) FTF. For each slice
        with constant 'axis' value a file is created in the (newly created)
        folder 'fname_stub'
        """
        if axis != 1 and axis != 0:
            raise ValueError("axis can only be 0 or 1 not " + str(axis))

        os.makedirs(fname_stub, exist_ok=overwrite)
        basepath, fname = os.path.split(fname_stub)
        path = os.path.join(basepath, fname, "")

        for i in np.arange(np.shape(self.X)[axis]):
            if axis == 0:
                p, x, y = self.run_operation(bp.cut, x_idx=i)
                label = self.metadata["xlabel"]
            elif axis == 1:
                x, p, y = self.run_operation(bp.cut, y_idx=i)
                label = self.metadata["ylabel"]

            param_fname = path + "/{fname}-{index:04d}-{label}={val}.dat"
            param_fname = param_fname.format(fname=fname, label=label,
                                             val=p[0,0], index=i)

            if os.path.isfile(param_fname) and not overwrite:
                raise(FileExistsError)

            x = np.squeeze(x)
            y = np.squeeze(y)
            if y.dtype == complex:
                data = np.vstack((x, np.zeros_like(x), np.real(y), np.imag(y)))
            else:
                data = np.vstack((x, y))

            header = """Field	field error	real part	imaginary part
(T)	(T)	(1)	(1)
µ\-(0)H	µ\-(0)H\-(err)	Real	Imag"""
            np.savetxt(param_fname, data.T, header=header, delimiter="\t")

# For compatibility reasons, we import the classes from the vna and timedomain
# submodule:
from bbFMR.measurement.vna import VNAMeasurement, VNAReferencedMeasurement
from bbFMR.measurement.timedomain import TimeDomainMeasurement