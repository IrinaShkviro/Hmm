# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:28:46 2015

@author: irka
"""

import gc

import numpy

import theano
import theano.tensor as T

from ichi_reader import ICHISeqDataReader

def zero_in_array(array):
    return [[0 for col in range(7)] for row in range(7)]
    
def train_da_sgd(learning_rate, window_size, training_epochs, corruption_level,
              train_set, da):
    
    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = da.input

    cost = da.get_cost(
        corruption_level=corruption_level
    )
    
    # compute the gradients of the cost of the `dA` with respect
    # to its parameters
    gparams = T.grad(cost, da.params)
    # generate the list of updates
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(da.params, gparams)
    ]
        
    train_da = theano.function(
        [index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set[index: index + window_size]
        }
    )
    
    ############
    # TRAINING #
    ############
    print(n_train_samples, 'train_samples')

    # go through training epochs
    while da.epoch < training_epochs:
        # go through trainng set
        cur_train_cost = []
        for index in xrange(n_train_samples):
            cur_train_cost.append(train_da(index))
        
        train_cost = float(numpy.mean(cur_train_cost))
        
        da.train_cost_array.append([])
        da.train_cost_array[-1].append(da.epoch)
        da.train_cost_array[-1].append(train_cost)
        cur_train_cost =[]
        
        da.epoch = da.epoch + 1

        print 'Training epoch %d, cost ' % da.epoch, train_cost

    return da
    
def pretraining_functions_sda_sgd(sda, window_size):
    ''' Generates a list of functions, each of them implementing one
    step in trainnig the dA corresponding to the layer with same index.
    The function will require as input the minibatch index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all minibatch indexes.

    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used
                        for training the dA

    :type window_size: int
    :param window_size: size of a window

    :type learning_rate: float
    :param learning_rate: learning rate used during training for any of
                              the dA layers
    '''

    index = T.lscalar('index')
    corruption_level = T.scalar('corruption')  # % of corruption to use
    learning_rate = T.scalar('lr')  # learning rate to use
    train_set = T.vector('train_set')

    pretrain_fns = []
    for cur_dA in sda.dA_layers:
        # get the cost and the updates list
        cost, updates = cur_dA.get_cost_updates(
            corruption_level=corruption_level,
            learning_rate=learning_rate
        )

        # compile the theano function
        fn = theano.function(
            inputs=[
                index,
                train_set,
                theano.Param(corruption_level, default=0.2),
                theano.Param(learning_rate, default=0.1)
            ],
            outputs=cost,
            updates=updates,
            givens={
                sda.x: train_set[index: index + window_size]
            }
        )
        # append `fn` to the list of functions
        pretrain_fns.append(fn)

    return pretrain_fns
    
def pretrain_sda_sgd(
        sda,
        train_names,
        read_window,
        read_algo,
        read_rank,
        window_size,
        pretraining_epochs,
        pretrain_lr,
        corruption_levels):
    # compute number of examples given in training set
    n_train_patients =  len(train_names)
    
    print '... getting the pretraining functions'
    pretraining_fns = pretraining_functions_sda_sgd(sda=sda,
                                                    window_size=window_size)

    print '... pre-training the model'
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        cur_dA = sda.dA_layers[i]
        cur_dA.train_cost_array = []
        cur_train_cost = []
        train_reader = ICHISeqDataReader(train_names)
        for patients in xrange(n_train_patients):
            # go through the training set
            train_set_x, train_set_y = train_reader.read_next_doc(
                algo = read_algo,
                window = read_window,
                rank = read_rank
            )
            n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
            cur_train_cost.append([])            
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                cur_epoch_cost=[]                               
                for index in xrange(n_train_samples):
                    cur_epoch_cost.append(pretraining_fns[i](index=index,
                             train_set = train_set_x.get_value(borrow=True),
                             corruption=corruption_levels[i],
                             lr=pretrain_lr))
                cur_train_cost[-1].append(numpy.mean(cur_epoch_cost))
            gc.collect()
            
        cur_dA.train_cost_array = [[epoch, cost] for epoch, cost in zip(xrange(pretraining_epochs), numpy.mean(cur_train_cost, axis=0))]
    return sda