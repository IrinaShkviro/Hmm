# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:14:22 2015

@author: irka
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 08:25:48 2015

@author: irka
"""

import os
import sys
import timeit
import gc

import numpy

import theano
import theano.tensor as T

from sklearn import hmm

from base import errors
#from MyVisualizer import visualize_hmm_for_one_label
from ichi_reader import ICHISeqDataReader

class GeneralHMM(object):
    """
    A general HMM model is a container with several Gaussian HMMs.
    We create and fit each of them and then estimate probability of appearance
    of the observations in each model, so we predict label as label of those
    HMM which probabity is the biggest.
    """

    def __init__(
        self,
        n_hiddens,
        n_hmms=7
    ):
        #create hmm models for each label
        self.n_hmms = n_hmms
        self.hmm_models = []
        for i in xrange(n_hmms):
            self.hmm_models.append(hmm.GaussianHMM(
                n_components=n_hiddens[i]
            ))
        self.valid_error_array=[]
            
    def train(self,
              train_names, valid_names,
              read_window,
              read_algo,
              read_rank,
              train_epochs
        ):
        for epoch in xrange(train_epochs):
            train_reader = ICHISeqDataReader(train_names)
            n_train_patients = len(train_names)
            #train hmms on data of each pattient
            for train_patient in xrange(n_train_patients):
                #get data divided on sequences with respect to labels
                train_set = train_reader.read_next_doc(
                    algo = read_algo,
                    rank = read_rank,
                    window = read_window,
                    divide = True
                )
                for label in xrange(self.n_hmms):
                    if train_set[label].eval() != []:
                        self.hmm_models[label].fit(
                            [numpy.array(train_set[label].eval()).reshape(-1, 1)]
                        )
                            
                error_cur_epoch = self.validate_model(
                    valid_names = valid_names,
                    read_window = read_window,
                    read_algo = read_algo,
                    read_rank = read_rank
                )
                self.valid_error_array.append([])
                self.valid_error_array[-1].append(epoch)
                self.valid_error_array[-1].append(train_patient)
                self.valid_error_array[-1].append(error_cur_epoch)
                
                gc.collect()
            
    def validate_model(self,
                       valid_names,
                       read_window,
                       read_algo,
                       read_rank
                       ):
        valid_reader = ICHISeqDataReader(valid_names)
        all_valid_x = []
        all_valid_y = []
        for i in xrange (len(valid_names)):
            valid_x = valid_reader.read_next_doc(
                algo = read_algo,
                rank = read_rank,
                window = read_window,
                divide = True
            )
            for label in xrange(self.n_hmms):
                d_x = valid_x[label].eval()
                all_valid_x = numpy.concatenate(
                    (all_valid_x, d_x)
                )
                all_valid_y = numpy.concatenate(
                    (all_valid_y, [label]*len(d_x))
                )
        #compute mean error value for patients in validation set
        error = mean_error(
            gen_hmm = self,
            obs_seq = all_valid_x,
            actual_states = all_valid_y
        )
        return error
            
    #compute label for one observation (with respect to window size)
    def define_label(self, obs):
        probabilities = [self.hmm_models[i].score(
                numpy.array([obs]).reshape((-1, 1))
            ) for i in xrange(self.n_hmms)]
        return numpy.argmax(probabilities)
        
    def define_labels_seq(self, obs_seq):
        return [self.define_label(obs_seq[i]) for i in xrange(len(obs_seq))]
    
def mean_error(gen_hmm, obs_seq, actual_states):
    predicted_states = gen_hmm.define_labels_seq(obs_seq)
    error_array=errors(
        predicted = numpy.array(predicted_states),
        actual = numpy.array(actual_states)
    )
    return error_array.eval().mean()
                        
def test_hmm(
    gen_hmm,
    test_names,
    read_window,
    read_algo,
    read_rank
    ):
    
    test_reader = ICHISeqDataReader(test_names)
    n_test_patients = len(test_names)
    
    error_array = []
    all_x = []
    all_y = []

    for i in xrange(n_test_patients):
        #get data divided on sequences with respect to labels
        test_x = test_reader.read_next_doc(
            algo = read_algo,
            rank = read_rank,
            window = read_window,
            divide = True
        )
        
        for label in xrange(gen_hmm.n_hmms):
            d_x = test_x[label].eval()
            all_x = numpy.concatenate(
                (all_x, d_x)
            )
            all_y = numpy.concatenate(
                (all_y, [label]*len(d_x))
            )
        
        #compute mean error value for one patient in test set
        patient_error = mean_error(
            gen_hmm = gen_hmm,
            obs_seq = all_x,
            actual_states = all_y
        )
        
        error_array.append(patient_error)
        print(patient_error, ' error for patient ' + str(test_names[i]))

        gc.collect()
    return error_array

def train_test_model():
    """train_data_names = ['p10a','p011','p013','p014','p020','p022','p040',
                        'p045','p048','p09b','p023','p035','p038', 'p09a','p033']
    valid_data_names = ['p09b','p023','p035','p038', 'p09a','p033']
    test_data_names = ['p002']"""
    without_train = ['p002','p003','p08a','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    train_data_names = ['p007', 'p028', 'p005', 'p08b']
    test_data_names = without_train
    valid_data_names = ['p09b', 'p021', 'p050']
    
    read_window = 20
    read_algo = 'filter'
    read_rank = 10
    
    train_epochs = 3
    
    hmms_count = 7    
    n_hiddens=[5]*hmms_count
    
    gen_hmm = GeneralHMM(
        n_hiddens = n_hiddens,
        n_hmms = hmms_count
    )
    
    #training
    gen_hmm.train(
        train_names = train_data_names,
        valid_names = valid_data_names,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        train_epochs = train_epochs
    )
    
    gc.collect()
    print('General HMM with several Gaussian hmms created')
    print('Start testing')
    
    error_array = test_hmm(
        gen_hmm = gen_hmm,
        test_names = test_data_names,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank
    )
    
    print(error_array)             
    print('mean value of error: ', numpy.round(numpy.mean(error_array), 6))
    print('min value of error: ', numpy.round(numpy.amin(error_array), 6))
    print('max value of error: ', numpy.round(numpy.amax(error_array), 6))
    
if __name__ == '__main__':
    train_test_model()