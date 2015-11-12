# -*- coding: utf-8 -*-
"""
Created on Fri Nov 06 23:11:17 2015

@author: irka
"""

import theano
import numpy
import os

import theano.tensor as T
import matplotlib.pyplot as plt

def errors(predicted, actual):
        """Return 1 if actual!=predicted (error) and 0 if right

        :type actual: theano.tensor.TensorType
        :param actual: corresponds to a vector that gives for each example the
                  correct label
        """

        actual = theano.shared(actual)
        predicted = theano.shared(predicted)
        # check if y has same dimension of y_pred
        if actual.ndim != predicted.ndim:
            raise TypeError(
                'actual should have the same shape as self.y_pred',
                ('actual', actual.type, 'y_pred', predicted.type)
            )

        # check if y is of the correct datatype
        if actual.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.neq(predicted, actual)
        else:
            raise NotImplementedError()

def errors2(predicted_states, actual_states, pat):
    """Return 1 if y!=y_predicted (error) and 0 if right

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
                  correct label
    """
    # check if y has same dimension of y_pred
    if predicted_states.ndim != actual_states.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('actual_states', actual_states.type, 'predicted_states', predicted_states.type)
        )
    
    predict = numpy.array([1
        if predicted_states[i] ==4
        else 0
        for i in xrange(predicted_states.shape[0])])
            
    actual = numpy.array([1
        if actual_states[i] ==4
        else 0
        for i in xrange(actual_states.shape[0])])
            
    base_folder = "temp_plots_for_2"
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    os.chdir(base_folder)
    # Make plots
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.plot(predict, 'b-')
    plt.plot(actual, 'r-',linewidth=2)
    plt.ylabel("z-coord")
    plt.legend(['predict','actual'])
    ax1.axes.get_xaxis().set_visible(False)
    plot_name = ('Patient [%s].png')%(pat)
    plt.savefig(plot_name, dpi=200)
    plt.close()
    os.chdir('../')

    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.neq(theano.shared(predict), theano.shared(actual))