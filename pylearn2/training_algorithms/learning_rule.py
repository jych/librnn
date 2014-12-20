"""
RMSProp
"""
__authors__ = ["Vincent Dumoulin", "Junyoung Chung"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = "Vincent Dumoulin"
__license__ = "3-clause BSD"
__maintainer__ = "Junyoung Chung"
__email__ = "chungjun@iro"

import numpy as np
from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.utils import sharedX

from pylearn2.training_algorithms.learning_rule import Momentum


class RMSPropMomentum(Momentum):
    """
    Implements the RMSprop

    Parameters
    ----------
    """

    def __init__(self,
                 init_momentum,
                 averaging_coeff=0.95,
                 stabilizer=1e-2,
                 use_first_order=False,
                 bound_inc=False,
                 momentum_clipping=None):
        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

        self.momentum_clipping = momentum_clipping
        if momentum_clipping is not None:
            self.momentum_clipping = np.cast[config.floatX](momentum_clipping)

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        velocity = OrderedDict()
        for param in grads.keys():

            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr +\
                (1 - self.averaging_coeff) * T.sqr(grads[param])
            if self.use_first_order:
                avg_grad = sharedX(np.zeros_like(param.get_value()))
                if param.name is not None:
                    avg_grad.name = 'avg_grad_' + param.name
                new_avg_grad = self.averaging_coeff * avg_grad +\
                    (1 - self.averaging_coeff) * grads[param]
                rms_grad_t = T.sqrt(new_avg_grad_sqr - new_avg_grad**2)
                updates[avg_grad] = new_avg_grad
            else:
                rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            normalized_grad = grads[param] / (rms_grad_t)
            new_velocity = self.momentum * velocity[param] -\
                learning_rate * normalized_grad

            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + new_velocity

        if self.momentum_clipping is not None:
            new_mom_norm = sum(
                map(lambda X: T.sqr(X).sum(),
                    [updates[velocity[param]] for param in grads.keys()])
            )
            new_mom_norm = T.sqrt(new_mom_norm)
            scaling_den = T.maximum(self.momentum_clipping, new_mom_norm)
            scaling_num = self.momentum_clipping

            for param in grads.keys():
                if self.bound_inc:
                    updates[velocity[param]] *= (scaling_num / scaling_den)
                    updates[param] = param + updates[velocity[param]]
                else:
                    updates[param] = param + updates[velocity[param]] *\
                        (scaling_num / scaling_den)

        return updates
