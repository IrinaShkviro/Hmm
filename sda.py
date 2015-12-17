# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:24:34 2015

@author: irka
"""

import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from hidden_layer import HiddenLayer
from da import dA
from visualizer import visualize_pretraining
from cg import pretrain_sda_cg
from sgd import pretrain_sda_sgd

theano.config.exception_verbosity='high'

class SdA(object):
    """Stacked denoising auto-encoder class (SdA)
    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        n_ins,
        hidden_layers_sizes,
        corruption_levels=[0.1, 0.1],
        theano_rng=None,
        n_outs=7
    ):
        """ This class is made to support a variable number of layers.
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`
        :type n_ins: int
        :param n_ins: dimension of the input to the sdA
        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value
        :type n_outs: int
        :param n_outs: dimension of the output of the network
        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.n_ins=n_ins
        self.n_outs=n_outs
        
        # allocate symbolic variables for the data
        self.x = T.vector('x')  # the data is presented as rasterized images
        self.y = T.iscalar('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(
                rng=numpy_rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.append(sigmoid_layer.theta)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(
                numpy_rng=numpy_rng,
                theano_rng=theano_rng,
                input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                theta=sigmoid_layer.theta
            )
            self.dA_layers.append(dA_layer)
        sda_input = T.vector('sda_input')  # the data is presented as rasterized images
        self.da_layers_output_size = hidden_layers_sizes[-1]
        self.get_da_output = theano.function(
            inputs=[sda_input],
            outputs=self.sigmoid_layers[-1].output.reshape((-1, self.da_layers_output_size)),
            givens={
                self.x: sda_input
            }
        )
        
    def set_classifier(self, classifier):
        self.classifier = classifier

def pretrain_SdA(train_names,
                 read_window,
                 read_algo,
                 read_rank,                 
                 window_size,
                 corruption_levels,
                 pretraining_epochs,
                 pretrain_lr,
                 pretrain_algo,
                 hidden_layers_sizes,
                 output_folder, base_folder):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.
    This is demonstrated on ICHI.
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer
    :type datasets: array
    :param datasets: [train_set, valid_set, test_set]
    
    :type output_folder: string
    :param output_folder: folder for costand error graphics with results
    """

    n_out = 7  # number of output units
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print('... building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=window_size,
        hidden_layers_sizes=hidden_layers_sizes,
        n_outs=n_out
    )
        
    #########################
    # PRETRAINING THE MODEL #
    #########################
    
    start_time = timeit.default_timer()
    
    if (pretrain_algo == "sgd"):
        pretrained_sda = pretrain_sda_sgd(
            sda=sda,
            train_names=train_names,
            read_window = read_window,
            read_algo = read_algo,
            read_rank = read_rank,
            window_size=window_size,
            pretraining_epochs=pretraining_epochs,
            pretrain_lr=pretrain_lr,
            corruption_levels=corruption_levels
        )
    else:
        pretrained_sda = pretrain_sda_cg(
            sda=sda,
            train_names=train_names,
            window_size=window_size,
            pretraining_epochs=pretraining_epochs,
            corruption_levels=corruption_levels,
            preprocess_algo = pretrain_algo,
            read_window = read_window
        )
                         
    end_time = timeit.default_timer()
    
    for i in xrange(sda.n_layers):
        print(i, 'i pretrained')
        visualize_pretraining(train_cost=pretrained_sda.dA_layers[i].train_cost_array,
                              window_size=window_size,
                              learning_rate=pretrain_lr,
                              corruption_level=corruption_levels[i],
                              n_hidden=sda.dA_layers[i].n_hidden,
                              da_layer=i,
                              datasets_folder=output_folder,
                              base_folder=base_folder)
    
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
   
    gc.collect()
    print('sda created')
    
    return sda
