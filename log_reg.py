"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.

"""
__docformat__ = 'restructedtext en'

import os
import sys
import time
import gc

import numpy

import theano
import theano.tensor as T

from ichi_reader import ICHISeqDataReader
from sgd import train_logistic_sgd

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.flatten(T.nnet.softmax(T.dot(input.reshape((1,n_in)),
                                                 self.W) + self.b))

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x)

        # parameters of the model
        self.params = [self.W, self.b]
        
        self.valid_error_array = []
        self.train_cost_array = []
        self.train_error_array = []
        
    def print_log_reg_types(self):
        print(self.W.type(), 'W')
        print(self.b.type(), 'b')
        print(self.p_y_given_x.type(), 'p_y_given_x')
        print(self.y_pred.type(), 'y_pred')
        

    def negative_log_likelihood(self, y):
        """Return the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a number that gives for each example the
                  correct label
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        
        #return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return -T.log(self.p_y_given_x)[y]
            
    def predict(self):
        """
        Return predicted y
        """
        return self.y_pred
        
    def distribution(self):
        return self.p_y_given_x
        
def zero_in_array(array):
    return [[0 for col in range(7)] for row in range(7)]

def train_log_reg(train_names,
                 valid_names,
                 read_algo,
                 read_window,
                 read_rank,
                 learning_rate,
                 n_epochs,
                 output_folder,
                 base_folder,
                 train_algo = 'sgd',
                 window_size=1):
    x = T.matrix('x')  # data, presented as window with x, y, x for each sample
    classifier = LogisticRegression(input=x, n_in=window_size*3, n_out=7)
    if (train_algo == 'sgd'):
        trained_classifier = train_logistic_sgd(
            read_algo = read_algo,
            read_window = read_window,
            read_rank = read_rank,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            train_names = train_names,
            valid_names = valid_names,
            classifier = classifier,
            window_size = window_size,
            output_folder = output_folder,
            base_folder = base_folder
        )
    #visualize log reg
    
    return trained_classifier
    
def test_log_reg(test_names,
                 read_algo,
                 read_window,
                 read_rank,                 
                 classifier,
                 window_size=1):
    test_reader = ICHISeqDataReader(test_names)
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.iscalar('y')
    
    test_error_array = []    
    
    for pat_num in xrange(len(test_names)):
        test_set_x, test_set_y = test_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
        n_test_samples = test_set_x.get_value(borrow=True).shape[0] - window_size + 1
        # compiling a Theano function that computes the mistakes that are made by
        # the model on a row
        test_model = theano.function(
            inputs=[index],
            outputs=[classifier.errors(y), classifier.predict(), y],
            givens={
                x: test_set_x[index: index + window_size],
                y: test_set_y[index + window_size - 1]
            }
        )
        
        test_result = [test_model(i) for i in xrange(n_test_samples)]
        test_result = numpy.asarray(test_result)
        test_losses = test_result[:,0]
        test_score = float(numpy.mean(test_losses))*100
                            
        test_error_array.append(test_score)
     
     return test_error_array
        
def test_all_params():  
    train_names = ['p002','p003','p005','p08a','p08b','p09a','p09b',
			'p10a','p011','p013','p014','p15a','p15b','p016',
               'p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p031','p032','p033',
               'p034','p035','p036','p038','p040','p042','p043',
               'p044','p045','p047','p048','p050','p051']
    valid_names = ['p017','p007','p012','p030','p037','p049']

    read_window = 1
    read_algo = 'avg'
    read_rank = 10
    
    learning_rate = 0.0001
    window_size = 50
    
    n_epochs = 15
    train_algo = 'sgd'

    error_array = []
    for test_pat in train_names:
        test_pat = train_names.pop(test_pat_num)
        output_folder=('all_data, [%s]')%(test_pat)
        trained_log_reg = train_log_reg(
            train_names = train_names,
            valid_names = valid_names,
            read_algo = read_algo,
            read_window = read_window,            
            read_rank = read_rank,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            train_algo = train_algo,
            window_size = window_size,
            output_folder = output_folder,
            base_folder = ('log_reg_%s')%(train_algo)
        )
        
        test_error = test_log_reg(
            test_names = [test_pat],
            read_algo = read_algo,
            read_window = read_window,
            read_rank = read_rank,                 
            classifier = trained_log_reg,
            window_size = window_size
        )[0]
        print ('error for patient %s is %f')%(test_pat, test_errors)
        error_array.append(test_error)
        train_names.insert(test_pat_num,test_pat)
    
    print(error_array)             
    print('mean value of error: ', numpy.round(numpy.mean(error_array), 6))
    print('min value of error: ', numpy.round(numpy.amin(error_array), 6))
    print('max value of error: ', numpy.round(numpy.amax(error_array), 6))    

if __name__ == '__main__':
    test_all_params()
