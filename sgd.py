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
from visualizer import visualize_finetuning

def zero_in_array(array):
    return [[0 for col in range(7)] for row in range(7)]
    
def training_functions_log_reg_sgd(classifier, window_size):
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

    # allocate symbolic variables for the data
    index = T.lscalar('index')

    # generate symbolic variables for input
    y = T.iscalar('y')  # labels, presented as int label
    train_set_x = T.vector('train_set_x')
    valid_set_x = T.vector('valid_set_x')
    train_set_y = T.ivector('train_set_y')
    valid_set_y = T.ivector('valid_set_y')
    
    cost = classifier.negative_log_likelihood(y)

    learning_rate = T.scalar('lr')  # learning rate to use
    
    # compute the gradient of cost with respect to theta = (W,b)
    g_theta = T.grad(cost=cost, wrt=classifier.theta)
    
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.theta, classifier.theta - learning_rate * g_theta)]
    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[
            index,
            train_set_x,
            train_set_y,
            theano.Param(learning_rate, default=0.001)
        ],
        outputs=[cost, classifier.errors(y), classifier.predict(), y],
        updates=updates,
        givens={
            classifier.x: train_set_x[index: index + window_size],
            y: train_set_y[index + window_size - 1]
        }
    )
    
    validate_model = theano.function(
        inputs=[
            index,
            valid_set_x,
            valid_set_y
        ],
        outputs=[classifier.errors(y), classifier.predict(), y],
        givens={
            classifier.x: valid_set_x[index: index + window_size],
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
    iter = 0
    classifier.train_cost_array = []
    classifier.train_error_array = []
    classifier.valid_error_array = []
    
    train_confusion_matrix = numpy.zeros((7, 7))
    valid_confusion_matrix = numpy.zeros((7, 7))
    
    for pat_num in xrange (len(train_names)):
        pat_epoch = 0
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
        
        train_model, validate_model = training_functions_log_reg_sgd(
            classifier = classifier,
            window_size = window_size
        )
        
        done_looping = False
        
        while (pat_epoch < n_epochs) and (not done_looping):
            cur_train_cost =[]
            cur_train_error = []
            train_confusion_matrix = zero_in_array(train_confusion_matrix)
            for index in xrange(n_train_samples):            
                sample_cost, sample_error, cur_pred, cur_actual = train_model(
                    index = index,
                    train_set_x = train_set_x.get_value(borrow=True),
                    train_set_y = train_set_y.eval(),
                    lr = learning_rate
                )
                # iteration number
                iter = pat_epoch * n_train_samples + index
                    
                cur_train_cost.append(sample_cost)
                cur_train_error.append(sample_error)
                train_confusion_matrix[cur_actual][cur_pred] += 1
            
                if (iter + 1) % validation_frequency == 0:
                    valid_confusion_matrix = zero_in_array(valid_confusion_matrix)
                    # compute zero-one loss on validation set
                    validation_losses = []
                    for i in xrange(n_valid_samples):
                        validation_loss, cur_pred, cur_actual = validate_model(
                            index = i,
                            valid_set_x = valid_set_x.get_value(borrow=True),
                            valid_set_y = valid_set_y.eval()
                        )
                        validation_losses.append(validation_loss)
                        valid_confusion_matrix[cur_actual][cur_pred] += 1
        
                    this_validation_loss = float(numpy.mean(validation_losses))*100                 
                    classifier.valid_error_array.append([])
                    classifier.valid_error_array[-1].append(classifier.epoch + float(iter)/n_train_samples)
                    classifier.valid_error_array[-1].append(this_validation_loss)
                            
                    print(
                        'epoch %i, iter %i/%i, validation error %f %%' %
                        (
                            classifier.epoch,
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
            classifier.train_cost_array[-1].append(classifier.epoch + float(iter)/n_train_samples)
            classifier.train_cost_array[-1].append(float(numpy.mean(cur_train_cost)))
            cur_train_cost =[]
           
            classifier.train_error_array.append([])
            classifier.train_error_array[-1].append(classifier.epoch + float(iter)/n_train_samples)
            classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
            cur_train_error =[]
                    
            classifier.epoch = classifier.epoch + 1
            pat_epoch = pat_epoch + 1
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
    valid_set = T.vector('valid_set')

    pretrain_fns = []
    valid_fns = []
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
        
        vf = theano.function(
            inputs=[
                index,
                valid_set,
                theano.Param(corruption_level, default=0.0)
            ],
            outputs=cost,
            givens={
                sda.x: valid_set[index: index + window_size]
            }
        )
                
        # append `fn` to the list of functions
        pretrain_fns.append(fn)
        valid_fns.append(vf)

    return pretrain_fns, valid_fns
    
def pretrain_sda_sgd(
        sda,
        train_names,
        valid_names,
        read_window,
        read_algo,
        read_rank,
        window_size,
        pretrain_lr,
        corruption_levels,
        global_epochs,
        pat_epochs):
    # compute number of examples given in training set
    n_train_patients =  len(train_names)
    
    print '... getting the pretraining functions'
    pretraining_fns, valid_fns = pretraining_functions_sda_sgd(sda=sda,
                                                    window_size=window_size)

    print '... pre-training the model'
    ## Pre-train layer-wise
    for i in xrange(sda.n_layers):
        cur_dA = sda.dA_layers[i]
        cur_dA.train_cost_array = []
        iter = 0
        for global_epoch in xrange(global_epochs):
            train_reader = ICHISeqDataReader(train_names)
            for patients in xrange(n_train_patients):
                # go through the training set
                train_set_x, train_set_y = train_reader.read_next_doc(
                    algo = read_algo,
                    window = read_window,
                    rank = read_rank
                )
                n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
                
                patience = n_train_samples*2  # look as this many examples regardless
                validation_frequency = patience / 2 # go through this many
                                          # minibatche before checking the network
                                          # on the validation set; in this case we
                                          # check every epoch
                # go through pretraining epochs
                for pat_epoch in xrange(pat_epochs):
                    cur_epoch_cost=[]                               
                    for index in xrange(n_train_samples):
                        # iteration number
                        big_epoch = (global_epoch*n_train_patients + patients)*pat_epochs + pat_epoch
                        iter = iter + 1
                    
                        cur_epoch_cost.append(pretraining_fns[i](index=index,
                                 train_set = train_set_x.get_value(borrow=True),
                                 corruption=corruption_levels[i],
                                 lr=pretrain_lr))
                                 
                        # test on valid set        
                        if (iter + 1) % validation_frequency == 0:
                            valid_reader = ICHISeqDataReader(valid_names)
                            valid_array = []
                            for valid_pat in xrange(len(valid_names)):
                                valid_set_x, valid_set_y = valid_reader.read_next_doc(
                                    algo = read_algo,
                                    window = read_window,
                                    rank = read_rank
                                )
                                n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1
                                validation_losses = [
                                    valid_fns[i](
                                        index = index,
                                        valid_set = valid_set_x.get_value(borrow=True)
                                    ) for index in xrange(n_valid_samples)]
                                this_validation_loss = float(numpy.mean(validation_losses))*100                 
                                valid_array.append(this_validation_loss)
                            valid_mean_error = numpy.mean(valid_array)                        
                            cur_dA.valid_error_array.append([])
                            cur_dA.valid_error_array[-1].append(
                                big_epoch + float(index)/n_train_samples
                            )
                            cur_dA.valid_error_array[-1].append(valid_mean_error)
                                        
                            print(
                                'epoch %i, iter %i/%i, validation error %f %%' %
                                (
                                    big_epoch,
                                    index + 1,
                                    n_train_samples,
                                    valid_mean_error
                                )
                            )
                    cur_dA.train_cost_array.append([])
                    cur_dA.train_cost_array[-1].append(big_epoch)
                    cur_dA.train_cost_array[-1].append(numpy.mean(cur_epoch_cost))
                    
                gc.collect()
            
    return sda

def build_finetune_functions(
        sda,
        window_size,
        learning_rate):
    '''Generates a function `train` that implements one step of
    finetuning, a function `validate` that computes the error on
    a batch from the validation set, and a function `test` that
    computes the error on a batch from the testing set
    :type datasets: list of pairs of theano.tensor.TensorType
    :param datasets: It is a list that contain all the datasets;
        the has to contain three pairs, `train`,
        `valid`, `test` in this order, where each pair
        is formed of two Theano variables, one for the
        datapoints, the other for the labels
    :type learning_rate: float
    :param learning_rate: learning rate used during finetune stage
    '''
    
    index = T.lscalar('index')  # index to a sample

    # compute the gradients with respect to the model parameters
    gparams = T.grad(sda.logLayer.negative_log_likelihood(sda.y), sda.params)

    # compute list of fine-tuning updates
    updates = [
        (param, param - gparam * learning_rate)
        for param, gparam in zip(sda.params, gparams)
    ]

    train_set_x = T.vector('train_set_x')
    valid_set_x = T.vector('valid_set_x')
    train_set_y = T.ivector('train_set_y')
    valid_set_y = T.ivector('valid_set_y')
    
    train_fn = theano.function(
        inputs=[
            index,
            train_set_x,
            train_set_y
        ],
        outputs=[sda.logLayer.negative_log_likelihood(sda.y),
                 sda.logLayer.errors(sda.y),
                 sda.logLayer.predict(),
                 sda.y],
        updates=updates,
        givens={
            sda.x: train_set_x[index: index + window_size],
            sda.y: train_set_y[index + window_size - 1]
        },
        name='train'
    )

    valid_score_i = theano.function(
        inputs = [
            index,
            valid_set_x,
            valid_set_y
        ],
        outputs=sda.logLayer.errors(sda.y),
        givens={
            sda.x: valid_set_x[index: index + window_size],
            sda.y: valid_set_y[index + window_size - 1]
        },
        name='valid'
    )

    return train_fn, valid_score_i

def finetune_log_layer_sgd(
    sda,
    train_names,
    valid_names,
    read_algo,
    read_window,
    read_rank,
    window_size,
    finetune_lr,
    global_epochs,
    pat_epochs,
    output_folder):
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing functions for the model
    print '... getting the finetuning functions'
    train_fn, validate_model = build_finetune_functions(
        sda=sda,
        window_size=window_size,
        learning_rate=finetune_lr
    )
    
    train_reader = ICHISeqDataReader(train_names)

    print '... finetunning the model'
    # early-stopping parameters
    patience_increase = 25  # wait this much longer when a new best is                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant   
                                  
    best_iter = 0
    best_valid = numpy.inf
    cur_train_cost =[]
    cur_train_error = []
    train_confusion_matrix = numpy.zeros((7, 7))
    iter = 0
    
    for global_epoch in xrange(global_epochs):
        for pat_num in xrange(len(train_names)):
            done_looping = False
            # go through the training set
            train_set_x, train_set_y = train_reader.read_next_doc(
                algo = read_algo,
                window = read_window,
                rank = read_rank
            )
            n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
            
            patience = n_train_samples*2  # look as this many examples regardless
            validation_frequency = patience / 2 # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch
            pat_epoch = 0
    
            while (pat_epoch < pat_epochs) and (not done_looping):
                train_confusion_matrix = zero_in_array(train_confusion_matrix)
                for index in xrange(n_train_samples):          
                    sample_cost, sample_error, cur_pred, cur_actual = train_fn(
                        index = index,
                        train_set_x = train_set_x.get_value(borrow=True),
                        train_set_y = train_set_y.eval()
                    )
                    
                    # iteration number
                    iter = iter + 1
                        
                    cur_train_cost.append(sample_cost)
                    cur_train_error.append(sample_error)
                    train_confusion_matrix[cur_actual][cur_pred] += 1
        
                    if (iter + 1) % validation_frequency == 0:
                        valid_reader = ICHISeqDataReader(valid_names)
                        valid_array = []
                        for valid_pat in xrange(len(valid_names)):
                            valid_set_x, valid_set_y = valid_reader.read_next_doc(
                                algo = read_algo,
                                window = read_window,
                                rank = read_rank
                            )
                            n_valid_samples = valid_set_x.get_value(borrow=True).shape[0] - window_size + 1
                            validation_losses = [
                                validate_model(
                                    index = i,
                                    valid_set_x = valid_set_x.get_value(borrow=True),
                                    valid_set_y = valid_set_y.eval()
                                ) for i in xrange(n_valid_samples)
                            ]
                            this_validation_loss = float(numpy.mean(validation_losses))*100                 
                            valid_array.append(this_validation_loss)
                        valid_mean_error = numpy.mean(valid_array)                        
                        sda.logLayer.valid_error_array.append([])
                        sda.logLayer.valid_error_array[-1].append(sda.logLayer.epoch + float(index)/n_train_samples)
                        sda.logLayer.valid_error_array[-1].append(valid_mean_error)
                                    
                        print(
                            'epoch %i, iter %i/%i, validation error %f %%' %
                            (
                                sda.logLayer.epoch,
                                iter + 1,
                                n_train_samples,
                                this_validation_loss
                            )
                        )
                        
                        # if we got the best validation score until now
                        if valid_mean_error < best_valid:
        
                            #improve patience if loss improvement is good enough
                            if this_validation_loss < best_valid * \
                                improvement_threshold:
                                patience = max(patience, iter * patience_increase)
        
                            best_iter = iter
                            best_valid = valid_mean_error
        
                    if patience*4 <= iter:
                        done_looping = True
                        print('Done looping')
                        break
                                   
                sda.logLayer.train_cost_array.append([])
                sda.logLayer.train_cost_array[-1].append(sda.logLayer.epoch)
                sda.logLayer.train_cost_array[-1].append(numpy.mean(cur_train_cost))
                cur_train_cost =[]
               
                sda.logLayer.train_error_array.append([])
                sda.logLayer.train_error_array[-1].append(sda.logLayer.epoch)
                sda.logLayer.train_error_array[-1].append(numpy.mean(cur_train_error)*100)
                cur_train_error =[]
                        
                sda.logLayer.epoch = sda.logLayer.epoch + 1
                pat_epoch = pat_epoch + 1
                gc.collect()
                            
            print(train_confusion_matrix, 'train_confusion_matrix')
            print(best_iter, 'best_iter')
    visualize_finetuning(
        train_cost=sda.logLayer.train_cost_array,
        train_error=sda.logLayer.train_error_array,
        valid_error=sda.logLayer.valid_error_array,
        window_size=window_size,
        learning_rate=0,
        datasets_folder=output_folder,
        base_folder='finetune_log_reg'
    )

    return sda
