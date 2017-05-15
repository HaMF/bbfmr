import matplotlib.pyplot as plt
import operator
from lmfit.model import _ensureMatplotlib, CompositeModel, Model, Parameters
class GlobalCompositeModel(CompositeModel):
    _names_collide = ("\nTwo models have parameters named '{clash}'. "
                      "Use distinct names.")
    _bad_arg   = "CompositeModel: argument {arg} is not a Model"

    def __init__(self, left, right, operator=operator.add, **kws):
        super().__init__(left, right, operator, **kws)

    def eval(self, params=None, **kwargs):
        return [vals[i] for i, vals in enumerate(self.eval_components(params=params, **kwargs).values())]

    @_ensureMatplotlib
    def plot(self, x, y=None, params=None, datafmt='o', fitfmt='-',
             initfmt='--', xlabel=None, ylabel=None, yerr=None,
             numpoints=None, fig=None, data_kws=None, fit_kws=None,
             init_kws=None, ax_res_kws=None, ax_fit_kws=None,
             fig_kws=None, labels=None):
        if fig is None:
            fig = plt.figure()
        if labels is None:
            labels = range(len(x))

        for i in range(len(x)):
            plt.plot(x[i], y[i], 's', label="data %s" % labels[i])
            plt.plot(x[i], self.components[i].eval(x=x[i], params=params),
                     label="best fit %s" % labels[i])
        plt.legend(loc="best")