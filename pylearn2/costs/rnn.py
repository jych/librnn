"""
Cost for Recurrent Neural Networks
"""
__authors__ = "Junyoung Chung"
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = "Junyoung Chung"
__license__ = "3-clause BSD"
__maintainer__ = "Junyoung Chung"
__email__ = "chungjun@iro"

import theano
import theano.tensor as T
from itertools import izip
from pylearn2.costs.cost import (
    Cost,
    DefaultDataSpecsMixin,
    NullDataSpecsMixin
)
from pylearn2.utils import safe_izip
from theano.compat.python2x import OrderedDict


class RNNCost(DefaultDataSpecsMixin, Cost):
    """
    The default cost defined in RNN class
    """
    supervised = True

    def __init__(self, gradient_clipping=False, max_magnitude=1):
        self.__dict__.update(locals())

    def expr(self, model, data, **kwargs):

        space, source = self.get_data_specs(model)
        space.validate(data)
        X, Y = data
        Y_hat = model.fprop(X)

        return T.cast(model.layers[-1].cost(Y, Y_hat), theano.config.floatX)

    def get_gradients(self, model, data, ** kwargs):

        cost = self.expr(model=model, data=data, **kwargs)

        params = list(model.get_params())

        grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))

        if self.gradient_clipping:
            norm_gs = 0.
            for grad in gradients.values():
                norm_gs += (grad ** 2).sum()
            not_finite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
            norm_gs = T.sqrt(norm_gs)
            norm_gs = T.switch(T.ge(norm_gs, self.max_magnitude),
                               self.max_magnitude / norm_gs,
                               1.)

            for param, grad in gradients.items():
                gradients[param] = T.switch(not_finite,
                                            .1 * param,
                                            grad * norm_gs)

        updates = OrderedDict()

        return gradients, updates


class WeightDecay(NullDataSpecsMixin, Cost):
    """
    A Cost that applies the following cost function:

    coeff * sum(sqr(weights))
    for each set of weights.

    Parameters
    ----------
    coeffs : list
        One element per layer, specifying the coefficient
        to put on the L1 activation cost for each layer.
        Each element may in turn be a list, ie, for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        self.get_data_specs(model)[0].validate(data)
        layer_costs = [layer.get_weight_decay(coeff)
                       for layer, coeff in safe_izip(model.layers,
                                                     self.coeffs)]

        assert T.scalar() != 0. # make sure theano semantics do what I want
        layer_costs = [cost for cost in layer_costs if cost != 0.]

        if len(layer_costs) == 0:
            rval = T.as_tensor_variable(0.)
            rval.name = '0_weight_decay'
            return rval
        else:
            total_cost = reduce(lambda x, y: x + y, layer_costs)
        total_cost.name = 'RNN_WeightDecay'

        assert total_cost.ndim == 0

        total_cost.name = 'weight_decay'

        return total_cost
