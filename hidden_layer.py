# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:31:33 2015

@author: irka
"""


__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, theta=None, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input.reshape((1,n_in))

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.

        if theta is None:
            theta = theano.shared(
                value=numpy.asarray(
                    rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_out + n_in + 1)),
                        high=4 * numpy.sqrt(6. / (n_out + n_in + 1)),
                        size=(n_in * n_out + n_out)
                    ),
                    dtype=theano.config.floatX
                ),
                name='theta',
                borrow=True
        )
        self.theta = theta
        
        # W is represented by the fisr n_visible*n_hidden elements of theta
        self.W = self.theta[0:n_in * n_out].reshape((n_in, n_out))
        # b is the rest (last n_hidden elements)
        self.b = self.theta[n_in * n_out:n_in * n_out + n_out]
        
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )