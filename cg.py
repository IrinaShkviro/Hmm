# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 11:13:42 2015

@author: irka
"""

import numpy

import theano
import theano.tensor as T
import scipy.optimize

from functools import partial

from ichi_reader import ICHISeqDataReader

def train_logistic_cg(
    read_algo,
    read_window,
    read_rank,
    train_names,
    valid_names,
    window_size,
    n_epochs,
    classifier):
    
    # read the datasets
    train_reader = ICHISeqDataReader(train_names)
    valid_reader = ICHISeqDataReader(valid_names)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  

    # generate symbolic variables for input
    x = classifier.x  # data, presented as window with x, y, x for each sample
    y = T.iscalar('y')  # labels, presented as int label


    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)
    
    for pat_num in xrange(len(train_names)):
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
        
        validate_model = theano.function(
            [index],
            classifier.errors(y),
            givens={
                x: valid_set_x[index: index + window_size],
                y: valid_set_y[index + window_size - 1]
            },
            name="validate"
        )
    
        #  compile a theano function that returns the cost
        conj_cost = theano.function(
            inputs=[index],
            outputs=[cost, classifier.errors(y), classifier.predict(), y],
            givens={
                x: train_set_x[index: index + window_size],
                y: train_set_y[index + window_size - 1]
            },
            name="conj_cost"
        )

        # compile a theano function that returns the gradient with respect to theta
        conj_grad = theano.function(
            [index],
            T.grad(cost, classifier.theta),
            givens={
                x: train_set_x[index: index + window_size],
                y: train_set_y[index + window_size - 1]
            },
            name="conj_grad"
        )
        
        train_confusion_matrix = numpy.zeros((7, 7))
    
        # creates a function that computes the average cost on the training set
        def train_fn(theta_value):
            classifier.theta.set_value(theta_value, borrow=True)
            cur_train_cost = []
            cur_train_error =[]
            for i in xrange(n_train_samples):
                sample_cost, sample_error, cur_pred, cur_actual = conj_cost(i)
                cur_train_cost.append(sample_cost)
                cur_train_error.append(sample_error)
                train_confusion_matrix[cur_actual][cur_pred] += 1
            
            this_train_loss = float(numpy.mean(cur_train_cost))  
            classifier.train_cost_array.append([])
            classifier.train_cost_array[-1].append(classifier.epoch)
            classifier.train_cost_array[-1].append(this_train_loss)
           
            classifier.train_error_array.append([])
            classifier.train_error_array[-1].append(classifier.epoch)
            classifier.train_error_array[-1].append(float(numpy.mean(cur_train_error)*100))
                    
            classifier.epoch += 1
            
            return this_train_loss
    
        # creates a function that computes the average gradient of cost with
        # respect to theta
        def train_fn_grad(theta_value):
            classifier.theta.set_value(theta_value, borrow=True)
            grad = conj_grad(0)
            for i in xrange(1, n_train_samples):
                grad += conj_grad(i)
            return grad / n_train_samples
    
        # creates the validation function
        def callback(theta_value):
            classifier.theta.set_value(theta_value, borrow=True)
            #compute the validation loss
            validation_losses = [validate_model(i)
                                 for i in xrange(n_valid_samples)]
            this_validation_loss = float(numpy.mean(validation_losses) * 100.,)
            print('validation error %f %%' % (this_validation_loss))
            classifier.valid_error_array.append([])
            classifier.valid_error_array[-1].append(classifier.epoch)
            classifier.valid_error_array[-1].append(this_validation_loss)
        
        ###############
        # TRAIN MODEL #
        ###############
    
        # using scipy conjugate gradient optimizer
        print ("Optimizing using scipy.optimize.fmin_cg...")
        best_theta = scipy.optimize.fmin_cg(
            f=train_fn,
            x0=numpy.zeros((classifier.n_in + 1) * classifier.n_out, dtype=x.dtype),
            fprime=train_fn_grad,
            callback=callback,
            disp=0,
            maxiter=n_epochs
        )
    return classifier
    
def train_da_cg(da, train_set, window_size, corruption_level, training_epochs):

    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1
    
     # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = da.input  # the data is presented as 3D vector

    cost = da.get_cost(
        corruption_level=corruption_level
    )
           
    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        da.theta.set_value(theta_value, borrow=True)
        
        #  compile a theano function that returns the cost
        conj_cost = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                x: train_set[index: index + window_size]
            },
            name="conj_cost"
        )
    
        train_losses = [conj_cost(i)
                        for i in xrange(n_train_samples)]
                            
        this_train_loss = float(numpy.mean(train_losses))  
        da.train_cost_array.append([])
        da.train_cost_array[-1].append(da.epoch)
        da.train_cost_array[-1].append(this_train_loss)
        da.epoch += 1
        return this_train_loss
        
    # creates a function that computes the average gradient of cost with
    # respect to theta
    def train_fn_grad(theta_value):
        da.theta.set_value(theta_value, borrow=True)
        
        # compile a theano function that returns the gradient with respect to theta
        conj_grad = theano.function(
            [index],
            T.grad(cost, da.theta),
            givens={
                x: train_set[index: index + window_size]
            },
            name="conj_grad"
        )
    
        grad = conj_grad(0)
        for i in xrange(1, n_train_samples):
            grad += conj_grad(i)
        return grad / n_train_samples

    ############
    # TRAINING #
    ############

    # using scipy conjugate gradient optimizer
    print ("Optimizing using scipy.optimize.fmin_cg...")
    best_w_b = scipy.optimize.fmin_cg(
        f=train_fn,
        x0=numpy.zeros((da.n_visible + 1) * da.n_hidden, dtype=x.dtype),
        fprime=train_fn_grad,
        disp=0,
        maxiter=training_epochs
    )
    return da

def pretraining_functions_sda_cg(sda, train_set_x, window_size, corruption_levels):
    ''' Generates a list of functions, each of them implementing one
    step in trainnig the dA corresponding to the layer with same index.
    The function will require as input the index, and to train
    a dA you just need to iterate, calling the corresponding function on
    all indexes.
    :type train_set_x: theano.tensor.TensorType
    :param train_set_x: Shared variable that contains all datapoints used        for training the dA
    :type window_size: int
    :param window_size: size of a window
    '''

    # index
    index = T.lscalar('index')
    #corruption_level = T.scalar('corruption')  # % of corruption to use
    n_train_samples = train_set_x.get_value(borrow=True).shape[0] - window_size + 1
            
    def train_fn(theta_value, da_index):
        cur_dA=sda.dA_layers[da_index]
        cost = cur_dA.get_cost(corruption_levels[da_index])
        
        # compile a theano function that returns the cost
        sample_cost = theano.function(
            inputs=[index],
            outputs=cost,
            givens={
                sda.x: train_set_x[index: index + window_size]
            },
            on_unused_input='warn'
        )
        
        sda.dA_layers[da_index].theta.set_value(theta_value, borrow=True)
        train_losses = [sample_cost(i)
                        for i in xrange(n_train_samples)]
        this_train_loss = float(numpy.mean(train_losses))  
        sda.dA_layers[da_index].train_cost_array.append([])
        sda.dA_layers[da_index].train_cost_array[-1].append(sda.dA_layers[da_index].epoch)
        sda.dA_layers[da_index].train_cost_array[-1].append(this_train_loss)
        sda.dA_layers[da_index].epoch += 1

        return numpy.mean(train_losses)
            
            
    def train_fn_grad(theta_value, da_index):
        cur_dA=sda.dA_layers[da_index]
        cost = cur_dA.get_cost(corruption_levels[da_index])
        
        # compile a theano function that returns the gradient with respect to theta
        sample_grad = theano.function(
            inputs=[index],
            outputs=T.grad(cost, cur_dA.theta),
            givens={
                sda.x: train_set_x[index: index + window_size]
            },
            on_unused_input='warn'
        )
        
        sda.dA_layers[da_index].theta.set_value(theta_value, borrow=True)
        grad = sample_grad(0)
        for i in xrange(1, n_train_samples):
            grad += sample_grad(i)
        return grad / n_train_samples
            
    return train_fn, train_fn_grad

def pretrain_sda_cg(sda, train_names, read_window, read_algo, read_rank, 
                    window_size, pretraining_epochs, corruption_levels):
    ## Pre-train layer-wise
    print '... getting the pretraining functions'
    import scipy.optimize
    
    for i in xrange(sda.n_layers):
        train_reader = ICHISeqDataReader(train_names)
        n_train_patients =  len(train_names)
        
        for patients in xrange(n_train_patients):
            train_set_x, train_set_y = train_reader.read_next_doc(
                algo = read_algo,
                window = read_window,
                rank = read_rank
            )
            pretraining_fn, pretraining_update = pretraining_functions_sda_cg(
                sda=sda,
                train_set_x=train_set_x,
                window_size=window_size,
                corruption_levels=corruption_levels
            )
            print '... pre-training the model'
            # using scipy conjugate gradient optimizer
            print ("Optimizing using scipy.optimize.fmin_cg...")
            best_w_b = scipy.optimize.fmin_cg(
                f=partial(pretraining_fn, da_index = i),
                x0=numpy.zeros((sda.dA_layers[i].n_visible + 1) * sda.dA_layers[i].n_hidden,
                               dtype=sda.dA_layers[i].input.dtype),
                fprime=partial(pretraining_update, da_index = i),
                maxiter=pretraining_epochs
            )                            
    return sda