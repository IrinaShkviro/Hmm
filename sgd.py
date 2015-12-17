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
    
def pretraining_functions_log_reg_sgd(classifier, window_size, datasets):
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

    # split the datasets
    (train_set_x, train_set_y) = datasets[0]
    (valid_set_x, valid_set_y) = datasets[1]

    # allocate symbolic variables for the data
    index = T.lscalar('index')

    # generate symbolic variables for input)
    x = classifier.input  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label

    cost = classifier.negative_log_likelihood(y)

    learning_rate = T.scalar('lr')  # learning rate to use
    
    # compute the gradient of cost with respect to theta = (W,b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=[cost, classifier.errors(y), classifier.predict(), y],
        updates=updates,
        givens={
            x: train_set_x[index: index + window_size],
            y: train_set_y[index + window_size - 1]
        }
    )
    
    validate_model = theano.function(
        inputs=[index],
        outputs=[classifier.errors(y), classifier.predict(), y],
        givens={
            x: valid_set_x[index: index + window_size],
            y: valid_set_y[index + window_size - 1]
        }
    )

    return train_model, validate_model

def train_logistic_sgd(
        read_algo,
        read_window,
        read_rank,
        learning_rate,
        n_epochs,
        train_names,
        valid_names,
        classifier,
        output_folder,
        base_folder,
        window_size=1
    ):
                          
    # read the datasets
    train_reader = ICHISeqDataReader(train_names)
    valid_reader = ICHISeqDataReader(valid_names)
    
    # early-stopping parameters    
    patience_increase = 25  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    best_validation_loss = numpy.inf

    done_looping = False
    epoch = 0
    iter = 0
    classifier.train_cost_array = []
    classifier.train_error_array = []
    classifier.valid_error_array = []
    
    train_confusion_matrix = numpy.zeros((7, 7))
    valid_confusion_matrix = numpy.zeros((7, 7))
    
    for pat_num in xrange (len(train_names)):
        # go through the training set
        train_set_x, train_set_y = train_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
        valid_set_x, valid_set_y = valid_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
        n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
        n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1
        
        patience = n_train_samples*2  # look as this many examples regardless
        validation_frequency = patience / 4
        
        train_model, validate_model = pretraining_functions_log_reg_sgd(
            classifier = classifier,
            window_size = window_size,
            datasets = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
        )
        
        while (epoch < n_epochs) and (not done_looping):
            cur_train_cost =[]
            cur_train_error = []
            train_confusion_matrix = zero_in_array(train_confusion_matrix)
            for index in xrange(n_train_samples):            
                sample_cost, sample_error, cur_pred, cur_actual = train_model(index)
                # iteration number
                iter = epoch * n_train_samples + index
                    
                cur_train_cost.append(sample_cost)
                cur_train_error.append(sample_error)
                train_confusion_matrix[cur_actual][cur_pred] += 1
            
                if (iter + 1) % validation_frequency == 0:
                    valid_confusion_matrix = zero_in_array(valid_confusion_matrix)
                    # compute zero-one loss on validation set
                    validation_losses = []
                    for i in xrange(n_valid_samples):
                        validation_loss, cur_pred, cur_actual = validate_model(i)
                        validation_losses.append(validation_loss)
                        valid_confusion_matrix[cur_actual][cur_pred] += 1
        
                    this_validation_loss = float(numpy.mean(validation_losses))*100                 
                    classifier.valid_error_array.append([])
                    classifier.valid_error_array[-1].append(pat_num)
                    classifier.valid_error_array[-1].append(float(iter)/n_train_samples)
                    classifier.valid_error_array[-1].append(this_validation_loss)
                            
                    print(
                        'epoch %i, iter %i/%i, validation error %f %%' %
                        (
                            epoch,
                            index + 1,
                            n_train_samples,
                            this_validation_loss
                        )
                    )
           
                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                            improvement_threshold:
                            patience = max(patience, iter * patience_increase)
            
                        best_validation_loss = this_validation_loss

                if patience*4 <= iter:
                    done_looping = True
                    print('Done looping')
                    break
                               
            classifier.train_cost_array.append([])
            classifier.train_cost_array[-1].append(pat_num)
            classifier.train_cost_array[-1].append(float(iter)/n_train_samples)
            classifier.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
            cur_train_cost =[]
           
            classifier.train_error_array.append([])
            classifier.train_error_array[-1].append(pat_num)
            classifier.train_error_array[-1].append(float(iter)/n_train_samples)
            classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
            cur_train_error =[]
                    
            epoch = epoch + 1
            gc.collect()
                        
        print(train_confusion_matrix, 'train_confusion_matrix')
        print(valid_confusion_matrix, 'valid_confusion_matrix')

    return classifier
    
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
