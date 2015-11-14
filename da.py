# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:48:49 2015

@author: irka
"""

"""
 This tutorial introduces denoising auto-encoders (dA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ichi_reader import ICHISeqDataReader
#from MyVisualizer import visualize_da
from sgd import train_da_sgd
from cg import train_da_cg

class dA(object):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space.
    """

    def __init__(
        self,
        numpy_rng,
        n_visible,
        n_hidden,
        theano_rng=None,
        input=None,
        theta=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input), the number of hidden units (the dimension
        d' of the latent or hidden space) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.input = input

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # initialize theta = (W,b) with 0s; W gets the shape (n_visible, n_hidden),
        # while b is a vector of n_out elements, making theta a vector of
        # n_visible*n_hidden + n_hidden elements
        if not theta:
            theta = theano.shared(
                value=numpy.asarray(
                    numpy_rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible + 1)),
                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible + 1)),
                        size=(n_visible * n_hidden + n_hidden)
                    ),
                    dtype=theano.config.floatX
                ),
                name='theta',
                borrow=True
            )
        self.theta = theta
        
        # W is represented by the fisr n_visible*n_hidden elements of theta
        W = self.theta[0:n_visible * n_hidden].reshape((n_visible, n_hidden))
        # b is the rest (last n_hidden elements)
        bhid = self.theta[n_visible * n_hidden:n_visible * n_hidden + n_hidden]

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    (self.n_visible,),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            self.x = T.vector(name='input')
        else:
            self.x = input

        self.params = [self.theta, self.b_prime]
        
        self.train_cost_array=[]
        self.epoch=0

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        return self.theano_rng.binomial(
            size=input.shape,
            n=1,
            p=1 - corruption_level,
            dtype=theano.config.floatX
        ) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost(self, corruption_level):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        
        y = self.get_hidden_values(tilde_x)
        self.z = self.get_reconstructed_input(y)
        
        cost = T.sqrt(T.sum(T.sqr(T.flatten(self.x - self.z))))
                
        return cost
        
    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        self.z = self.get_reconstructed_input(y)
        cost = T.sqrt(T.sum(T.sqr(T.flatten(self.x - self.z, outdim=1))))

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
        
    def predict(self):
        """
        Return predicted vector
        """
        return self.z
    
    def actual(self):
        """
        Return actual vector
        """
        return self.x    
        
def train_dA(
    learning_rate,
    training_epochs,
    window_size,
    corruption_level,
    n_hidden,
    train_set,
    output_folder,
    train_algo="sgd"):

    """
    This dA is tested on ICHI_Data

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type window_size: int
    :param window_size: size of window used for training

    :type corruption_level: float
    :param corruption_level: corruption_level used for training the DeNosing
                          AutoEncoder

    :type n_hidden: int
    :param n_hidden: count of nodes in hidden layer

    :type output_folder: string
    :param output_folder: folder for costand error graphics with results

    """
    
    start_time = time.clock()
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    
    x = T.vector('x')    

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=window_size,
        n_hidden=n_hidden
    )
    
    if train_algo == "sgd":
        updated_da = train_da_sgd(
            learning_rate=learning_rate,
            window_size=window_size,
            training_epochs=training_epochs,
            corruption_level=corruption_level,
            train_set=train_set,
            da=da
        )
        base_folder = "da_sgd"
    else:
        updated_da = train_da_cg(
            da=da,
            train_set=train_set,
            window_size=window_size,
            corruption_level=corruption_level,
            training_epochs=training_epochs
        )
        base_folder = "da_cg"
    
    visualize_da(train_cost=updated_da.train_cost_array,
                 window_size=window_size,
                 learning_rate=learning_rate,
                 corruption_level=corruption_level,
                 n_hidden=n_hidden,
                 output_folder=output_folder,
                 base_folder=base_folder)
    
    end_time = time.clock()
    training_time = (end_time - start_time)
    
    print >> sys.stderr, ('The with corruption %f code for file ' +
                          os.path.split(__file__)[1] +
                         ' ran for %.2fm' % (corruption_level, (training_time) / 60.))

def test_da_params(corruption_level):
    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.013, 0.015]
    window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    
    train_data = ['p10a','p011','p013','p014','p020','p022','p040','p045','p048']
    valid_data = ['p09b','p023','p035','p038']
    test_data = ['p09a','p033']
    
    train_reader = ICHISeqDataReader(train_data)
    train_set, train_labels = train_reader.read_all()
    
    valid_reader = ICHISeqDataReader(valid_data)
    valid_set, valid_labels = valid_reader.read_all()

    test_reader = ICHISeqDataReader(test_data)
    test_set, test_labels = test_reader.read_all()
    
    output_folder=('[%s], [%s], [%s]')%(",".join(train_data), ",".join(valid_data), ",".join(test_data))
    
    for lr in learning_rates:
        for ws in window_sizes:
            train_dA(learning_rate=lr,
                     training_epochs=1,
                     window_size = ws, 
                     corruption_level=corruption_level,
                     n_hidden=ws*2,
                     train_set=train_set,
                     output_folder=output_folder)


if __name__ == '__main__':
    test_da_params(corruption_level=0.)
    test_da_params(corruption_level=0.3)