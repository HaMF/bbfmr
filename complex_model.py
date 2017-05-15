# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:07:02 2016

@author: hannes.maierflaig
"""

import operator
from lmfit.model import Model, Parameter, _align, _ensureMatplotlib, warnings
from lmfit.model import ModelResult as ModelResultBase
from lmfit.minimizer import Minimizer
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


class ComplexModelResult(ModelResultBase):
    def __init__(self, model, params, data=None, weights=None,
                 method='leastsq', fcn_args=None, fcn_kws=None,
                 iter_cb=None, scale_covar=True, **fit_kws):
        self.complex_data = None
        self.model = model
        self.data = data
        self.weights = weights
        self.method = method
        self.ci_out = None
        self.init_params = deepcopy(params)

        # modify residual fcnt here in order to get an array of real floats
        def reim_residual(*a, **kws):
            res = model._residual(*a, **kws)
            return res.view(np.float)

        Minimizer.__init__(self, reim_residual, params, fcn_args=fcn_args,
                           fcn_kws=fcn_kws, iter_cb=iter_cb,
                           scale_covar=scale_covar, **fit_kws)

    @_ensureMatplotlib
    def plot_fit(self, ax=None, datafmt='o', fitfmt='-', initfmt='--', yerr=None,
                 numpoints=None,  data_kws=None, fit_kws=None, init_kws=None,
                 ax_kws=None, norm=lambda x: x):
        """Plot the fit results using matplotlib.

        The method will plot results of the fit using matplotlib, including:
        the data points, the initial fit curve and the fitted curve. If the fit
        model included weights, errorbars will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : string, optional
            matplotlib format string for data points
        fitfmt : string, optional
            matplotlib format string for fitted curve
        initfmt : string, optional
            matplotlib format string for initial conditions for the fit
        yerr : ndarray, optional
            array of uncertainties for data array
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        init_kws : dictionary, optional
            keyword arguments passed on to the plot function for the initial
            conditions of the fit
        ax_kws : dictionary, optional
            keyword arguments for a new axis, if there is one being created

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        ----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `ax` is None then matplotlib.pyplot.gca(**ax_kws) is called.

        See Also
        --------
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_kws is None:
            ax_kws = {}

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        x_array = self.userkws[independent_var]

        # make a dense array for x-axis if data is not dense
        if numpoints is not None and len(self.data) < numpoints:
            x_array_dense = np.linspace(min(x_array), max(x_array), numpoints)
        else:
            x_array_dense = x_array

        ax.plot(x_array_dense, norm(self.model.eval(self.init_params,
                **{independent_var: x_array_dense})), initfmt,
                label='init', **init_kws)
        ax.plot(x_array_dense, norm(self.model.eval(self.params,
                **{independent_var: x_array_dense})), fitfmt,
                label='best-fit', **fit_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights
        if yerr is not None:
            ax.errorbar(x_array, norm(self.data), yerr=norm(yerr),
                        fmt=datafmt, label='data', **data_kws)
        else:
            ax.plot(x_array, norm(self.data), datafmt, label='data',
                    **data_kws)

        ax.set_title(self.model.name)
        ax.set_xlabel(independent_var)
        ax.set_ylabel('y')
        ax.legend()

        return ax

    @_ensureMatplotlib
    def plot_residuals(self, ax=None, datafmt='o', yerr=None, data_kws=None,
                       fit_kws=None, ax_kws=None, norm=lambda x: x):
        """Plot the fit residuals using matplotlib.

        The method will plot residuals of the fit using matplotlib, including:
        the data points and the fitted curve (as horizontal line). If the fit
        model included weights, errorbars will also be plotted.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. The default in None, which means use the
            current pyplot axis or create one if there is none.
        datafmt : string, optional
            matplotlib format string for data points
        yerr : ndarray, optional
            array of uncertainties for data array
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        ax_kws : dictionary, optional
            keyword arguments for a new axis, if there is one being created

        Returns
        -------
        matplotlib.axes.Axes

        Notes
        ----
        For details about plot format strings and keyword arguments see
        documentation of matplotlib.axes.Axes.plot.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `ax` is None then matplotlib.pyplot.gca(**ax_kws) is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot : Plot the fit results and residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if ax_kws is None:
            ax_kws = {}

        if len(self.model.independent_vars) == 1:
            independent_var = self.model.independent_vars[0]
        else:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(ax, plt.Axes):
            ax = plt.gca(**ax_kws)

        x_array = self.userkws[independent_var]

        ax.axhline(0, **fit_kws)

        if yerr is None and self.weights is not None:
            yerr = 1.0/self.weights
        if yerr is not None:
            ax.errorbar(x_array, norm(self.eval() - self.data),
                        yerr=norm(yerr), fmt=datafmt, label='residuals',
                        **data_kws)
        else:
            ax.plot(x_array, norm(self.eval() - self.data), datafmt,
                    label='residuals', **data_kws)

        ax.set_title(self.model.name)
        ax.set_ylabel('residuals')
        ax.legend()

        return ax

    @_ensureMatplotlib
    def plot(self, datafmt='o', fitfmt='-', initfmt='--', yerr=None,
             numpoints=None, fig=None, data_kws=None, fit_kws=None,
             init_kws=None, ax_res_kws=None, ax_fit_kws=None, fig_kws=None):
        """Plot the fit results and residuals using matplotlib.

        The method will produce a matplotlib figure with both results of the
        fit and the residuals plotted. If the fit model included weights,
        errorbars will also be plotted.

        Parameters
        ----------
        datafmt : string, optional
            matplotlib format string for data points
        fitfmt : string, optional
            matplotlib format string for fitted curve
        initfmt : string, optional
            matplotlib format string for initial conditions for the fit
        yerr : ndarray, optional
            array of uncertainties for data array
        numpoints : int, optional
            If provided, the final and initial fit curves are evaluated not
            only at data points, but refined to contain `numpoints` points in
            total.
        fig : matplotlib.figure.Figure, optional
            The figure to plot on. The default in None, which means use the
            current pyplot figure or create one if there is none.
        data_kws : dictionary, optional
            keyword arguments passed on to the plot function for data points
        fit_kws : dictionary, optional
            keyword arguments passed on to the plot function for fitted curve
        init_kws : dictionary, optional
            keyword arguments passed on to the plot function for the initial
            conditions of the fit
        ax_res_kws : dictionary, optional
            keyword arguments for the axes for the residuals plot
        ax_fit_kws : dictionary, optional
            keyword arguments for the axes for the fit plot
        fig_kws : dictionary, optional
            keyword arguments for a new figure, if there is one being created

        Returns
        -------
        matplotlib.figure.Figure

        Notes
        ----
        The method combines ModelResult.plot_fit and ModelResult.plot_residuals.

        If yerr is specified or if the fit model included weights, then
        matplotlib.axes.Axes.errorbar is used to plot the data.  If yerr is
        not specified and the fit includes weights, yerr set to 1/self.weights

        If `fig` is None then matplotlib.pyplot.figure(**fig_kws) is called.

        See Also
        --------
        ModelResult.plot_fit : Plot the fit results using matplotlib.
        ModelResult.plot_residuals : Plot the fit residuals using matplotlib.
        """
        if data_kws is None:
            data_kws = {}
        if fit_kws is None:
            fit_kws = {}
        if init_kws is None:
            init_kws = {}
        if ax_res_kws is None:
            ax_res_kws = {}
        if ax_fit_kws is None:
            ax_fit_kws = {}
        if fig_kws is None:
            fig_kws = {}

        if len(self.model.independent_vars) != 1:
            print('Fit can only be plotted if the model function has one '
                  'independent variable.')
            return False

        if not isinstance(fig, plt.Figure):
            fig = plt.figure(**fig_kws)

        if self.data.dtype == complex:
            ncols = 2
            gs = plt.GridSpec(nrows=2, ncols=ncols, height_ratios=[1, 4])
            for i, norm in enumerate((np.real, np.imag)):
                ax_res = fig.add_subplot(gs[0, i], **ax_res_kws)
                ax_fit = fig.add_subplot(gs[1, i], sharex=ax_res, **ax_fit_kws)
                self.plot_fit(ax=ax_fit, datafmt=datafmt, fitfmt=fitfmt, yerr=yerr,
                              initfmt=initfmt, numpoints=numpoints, data_kws=data_kws,
                              fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws,
                              norm=norm)
                self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr,
                                    data_kws=data_kws, fit_kws=fit_kws,
                                    ax_kws=ax_res_kws, norm=norm)
        else:
            ncols = 1
            gs = plt.GridSpec(nrows=2, ncols=ncols, height_ratios=[1, 4])
            ax_res = fig.add_subplot(gs[0, 0], **ax_res_kws)
            ax_fit = fig.add_subplot(gs[1, 0], sharex=ax_res, **ax_fit_kws)

            self.plot_fit(ax=ax_fit, datafmt=datafmt, fitfmt=fitfmt, yerr=yerr,
                          initfmt=initfmt, numpoints=numpoints, data_kws=data_kws,
                          fit_kws=fit_kws, init_kws=init_kws, ax_kws=ax_fit_kws)
            self.plot_residuals(ax=ax_res, datafmt=datafmt, yerr=yerr,
                                data_kws=data_kws, fit_kws=fit_kws,
                                ax_kws=ax_res_kws)

        return fig


class ComplexModel(Model):
    def __init__(self, func, **kwargs):
        # initialize model, find parameter names etc.
        super().__init__(func, **kwargs)

    def fit(self, data, params=None, weights=None, method='leastsq',
            iter_cb=None, scale_covar=True, verbose=True, fit_kws=None,
            **kwargs):
        """Fit the model to the data.

        Parameters
        ----------
        data: array-like
        params: Parameters object
        weights: array-like of same size as data
            used for weighted fit
        method: fitting method to use (default = 'leastsq')
        iter_cb:  None or callable  callback function to call at each iteration.
        scale_covar:  bool (default True) whether to auto-scale covariance matrix
        verbose: bool (default True) print a message when a new parameter is
            added because of a hint.
        fit_kws: dict
            default fitting options, such as xtol and maxfev, for scipy optimizer
        keyword arguments: optional, named like the arguments of the
            model function, will override params. See examples below.

        Returns
        -------
        lmfit.ModelResult

        Examples
        --------
        # Take t to be the independent variable and data to be the
        # curve we will fit.

        # Using keyword arguments to set initial guesses
        >>> result = my_model.fit(data, tau=5, N=3, t=t)

        # Or, for more control, pass a Parameters object.
        >>> result = my_model.fit(data, params, t=t)

        # Keyword arguments override Parameters.
        >>> result = my_model.fit(data, params, tau=5, t=t)

        Note
        ----
        All parameters, however passed, are copied on input, so the original
        Parameter objects are unchanged.

        """
        if params is None:
            params = self.make_params(verbose=verbose)
        else:
            params = deepcopy(params)

        # If any kwargs match parameter names, override params.
        param_kwargs = set(kwargs.keys()) & set(self.param_names)
        for name in param_kwargs:
            p = kwargs[name]
            if isinstance(p, Parameter):
                p.name = name  # allows N=Parameter(value=5) with implicit name
                params[name] = deepcopy(p)
            else:
                params[name].set(value=p)
            del kwargs[name]

        # All remaining kwargs should correspond to independent variables.
        for name in kwargs.keys():
            if name not in self.independent_vars:
                warnings.warn("The keyword argument %s does not" % name +
                              "match any arguments of the model function." +
                              "It will be ignored.", UserWarning)

        # If any parameter is not initialized raise a more helpful error.
        missing_param = any([p not in params.keys()
                             for p in self.param_names])
        blank_param = any([(p.value is None and p.expr is None)
                           for p in params.values()])
        if missing_param or blank_param:
            msg = ('Assign each parameter an initial value by passing '
                   'Parameters or keyword arguments to fit.\n')
            missing = [p for p in self.param_names if p not in params.keys()]
            blank = [name for name, p in params.items()
                                    if (p.value is None and p.expr is None)]
            msg += 'Missing parameters: %s\n' % str(missing)
            msg += 'Non initialized parameters: %s' % str(blank)
            raise ValueError(msg)

        # Do not alter anything that implements the array interface (np.array, pd.Series)
        # but convert other iterables (e.g., Python lists) to numpy arrays.
        if not hasattr(data, '__array__'):
            data = np.asarray(data)
        for var in self.independent_vars:
            var_data = kwargs[var]
            if (not hasattr(var_data, '__array__')) and (not np.isscalar(var_data)):
                kwargs[var] = np.asfarray(var_data)

        # Handle null/missing values.
        mask = None
        if self.missing not in (None, 'none'):
            mask = self._handle_missing(data)  # This can raise.
            if mask is not None:
                data = data[mask]
            if weights is not None:
                weights = _align(weights, mask, data)

        # If independent_vars and data are alignable (pandas), align them,
        # and apply the mask from above if there is one.
        for var in self.independent_vars:
            if not np.isscalar(kwargs[var]):
                kwargs[var] = _align(kwargs[var], mask, data)

        if fit_kws is None:
            fit_kws = {}

        output = ComplexModelResult(self, params, method=method,
                                    iter_cb=iter_cb, scale_covar=scale_covar,
                                    fcn_kws=kwargs, **fit_kws)
        output.fit(data=data, weights=weights)
        output.components = self.components
        return output

    def __add__(self, other):
        return ComplexCompositeModel(self, other, operator.add)

    def __sub__(self, other):
        return ComplexCompositeModel(self, other, operator.sub)

    def __mul__(self, other):
        return ComplexCompositeModel(self, other, operator.mul)

    def __div__(self, other):
        return ComplexCompositeModel(self, other, operator.truediv)

    def __truediv__(self, other):
        return ComplexCompositeModel(self, other, operator.truediv)


class ComplexCompositeModel(ComplexModel):
    _names_collide = ("\nTwo models have parameters named '{clash}'. "
                      "Use distinct names.")
    _bad_arg   = "CompositeModel: argument {arg} is not a Model"
    _bad_op    = "CompositeModel: operator {op} is not callable"
    _known_ops = {operator.add: '+', operator.sub: '-',
                  operator.mul: '*', operator.truediv: '/'}

    def __init__(self, left, right, op, **kws):
        if not isinstance(left, Model):
            raise ValueError(self._bad_arg.format(arg=left))
        if not isinstance(right, Model):
            raise ValueError(self._bad_arg.format(arg=right))
        if not callable(op):
            raise ValueError(self._bad_op.format(op=op))

        self.left  = left
        self.right = right
        self.op    = op

        name_collisions = set(left.param_names) & set(right.param_names)
        if len(name_collisions) > 0:
            msg = ''
            for collision in name_collisions:
                msg += self._names_collide.format(clash=collision)
            raise NameError(msg)

        # we assume that all the sub-models have the same independent vars
        if 'independent_vars' not in kws:
            kws['independent_vars'] = self.left.independent_vars
        if 'missing' not in kws:
            kws['missing'] = self.left.missing

        def _tmp(*args, **kws): pass
        ComplexModel.__init__(self, _tmp, **kws)

        for side in (left, right):
            prefix = side.prefix
            for basename, hint in side.param_hints.items():
                self.param_hints["%s%s" % (prefix, basename)] = hint

    def _parse_params(self):
        self._func_haskeywords = (self.left._func_haskeywords or
                                  self.right._func_haskeywords)
        self._func_allargs = (self.left._func_allargs +
                              self.right._func_allargs)
        self.def_vals = deepcopy(self.right.def_vals)
        self.def_vals.update(self.left.def_vals)
        self.opts = deepcopy(self.right.opts)
        self.opts.update(self.left.opts)

    def _reprstring(self, long=False):
        return "(%s %s %s)" % (self.left._reprstring(long=long),
                               self._known_ops.get(self.op, self.op),
                               self.right._reprstring(long=long))

    def eval(self, params=None, **kwargs):
        return self.op(self.left.eval(params=params, **kwargs),
                       self.right.eval(params=params, **kwargs))

    def eval_components(self, **kwargs):
        """return ordered dict of name, results for each component"""
        out = OrderedDict(self.left.eval_components(**kwargs))
        out.update(self.right.eval_components(**kwargs))
        return out

    @property
    def param_names(self):
        return self.left.param_names + self.right.param_names

    @property
    def components(self):
        """return components for composite model"""
        return self.left.components + self.right.components

    def _make_all_args(self, params=None, **kwargs):
        """generate **all** function args for all functions"""
        out = self.right._make_all_args(params=params, **kwargs)
        out.update(self.left._make_all_args(params=params, **kwargs))
        return out
