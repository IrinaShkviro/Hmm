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

def train_da_cg(da, train_set, window_size, corruption_level, training_epochs):

    n_train_samples = train_set.get_value(borrow=True).shape[0] - window_size + 1
    
     # allocate symbolic variables for the data
    index = T.lscalar()    # index
    x = da.input  # the data is presented as 3D vector

    cost = da.get_cost(
        corruption_level=corruption_level
    )
           
    #  compile a theano function that returns the cost
    conj_cost = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set[index: index + window_size]
        },
        name="conj_cost"
    )

    # compile a theano function that returns the gradient with respect to theta
    conj_grad = theano.function(
        [index],
        T.grad(cost, da.theta),
        givens={
            x: train_set[index: index + window_size]
        },
        name="conj_grad"
    )
    
    # creates a function that computes the average cost on the training set
    def train_fn(theta_value):
        da.theta.set_value(theta_value, borrow=True)
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
    
def pretrain_sda_cg(sda, train_names, read_window, window_size,
                    preprocess_algo, pretraining_epochs, corruption_levels):
    ## Pre-train layer-wise
    print '... getting the pretraining functions'
    import scipy.optimize
    train_reader = ICHISeqDataReader(train_names)
    n_train_patients =  len(train_names)
    
    for patients in xrange(n_train_patients):
        train_set_x, train_set_y = train_reader.read_next_doc(
            preprocess_algo = preprocess_algo,
            window_size = read_window
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
        for i in xrange(sda.n_layers):
            best_w_b = scipy.optimize.fmin_cg(
                f=partial(pretraining_fn, da_index = i),
                x0=numpy.zeros((sda.dA_layers[i].n_visible + 1) * sda.dA_layers[i].n_hidden,
                               dtype=sda.dA_layers[i].input.dtype),
                fprime=partial(pretraining_update, da_index = i),
                maxiter=pretraining_epochs
            )                            
    return sda