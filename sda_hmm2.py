# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 08:21:30 2015

@author: irka
"""

import os
import sys
import timeit
import gc

import numpy
from sklearn import hmm

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from sda import SdA, pretrain_SdA
from logistic import LogisticRegression
from mlp import HiddenLayer
from dA import dA
from MyVisualizer import visualize_pretraining, visualize_finetuning
from ichi_seq_data_reader import ICHISeqDataReader
from cg import pretrain_sda_cg, finetune_sda_cg
from sgd import pretrain_sda_sgd, finetune_sda_sgd
from HMM_second_with_sklearn import update_params_on_patient,\
 finish_training, get_error_on_patient, get_error_on_patient2
from preprocess import preprocess_av_disp

theano.config.exception_verbosity='high'

def train_SdA(train_names,
             output_folder, base_folder,
             window_size,
             corruption_levels,
             pretraining_epochs,
             rank,
             pretrain_lr,
             pretrain_algo,
             hidden_layers_sizes,
             read_window = 1):
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
 
    #########################
    # PRETRAINING THE MODEL #
    #########################
       
    pretrained_sda = retrained_SdA(
        train_names = train_names,
        output_folder = output_folder,
        base_folder = base_folder,
        window_size = window_size,
        corruption_levels = corruption_levels,
        pretraining_epochs,
        pretrain_lr = pretraining_epochs,
        pretrain_algo = pretrain_algo,
        hidden_layers_sizes = hidden_layers_sizes
    )
        
    
    for i in xrange(sda.n_layers):
        print(i, 'i pretrained')
        visualize_pretraining(train_cost=pretrained_sda.dA_layers[i].train_cost_array,
                              window_size=window_size,
                              learning_rate=0,
                              corruption_level=corruption_levels[i],
                              n_hidden=sda.dA_layers[i].n_hidden,
                              da_layer=i,
                              datasets_folder=output_folder,
                              base_folder=base_folder)

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    n_train_patients=len(train_names)
    
    n_visible=pow(10, rank) + 2 - window_size
    n_hidden=n_out
        
    train_reader = ICHISeqDataReader(train_names)
    
    #create matrices for params of HMM layer
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden))
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))
    
    preprocess_algo = "filter"
    if (preprocess_algo == "avg_disp" or preprocess_algo == "filter+avg_disp"):
        n_visible *= 10

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_next_doc(
            algo = preprocess_algo,
            window = read_window
        )
        train_set_x = train_set_x.get_value()
        train_set_y = train_set_y.eval()
        n_train_times = train_set_x.shape[0] - window_size + 1
        
        train_visible_after_sda = numpy.array([sda.get_da_output(
                train_set_x[time: time+window_size]).ravel()
                for time in xrange(n_train_times)])
                    
        new_train_visible = create_labels_after_das(
            da_output_matrix=train_visible_after_sda,
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        n_patient_samples = len(train_set_y)
        half_window_size = int(window_size/2)
        new_train_hidden=train_set_y[half_window_size:n_patient_samples-half_window_size]
        
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=new_train_hidden,
            visibles_patient=new_train_visible,
            n_hidden=n_hidden
        )
        
        gc.collect()
        
    pi_values, a_values, b_values = finish_training(
        pi_values=pi_values,
        a_values=a_values,
        b_values=b_values,
        array_from_hidden=array_from_hidden,
        n_hidden=n_hidden,
        n_patients=n_train_patients
    )
    
    hmm_model = hmm.MultinomialHMM(
        n_components=n_hidden,
        startprob=pi_values,
        transmat=a_values
    )
    
    hmm_model.n_symbols=n_visible
    hmm_model.emissionprob_=b_values 
    gc.collect()
    print('MultinomialHMM created')
    
    sda.set_hmm_layer(
        hmm_model=hmm_model
    )
    return sda
    
def test_sda(sda, test_names, rank, start_base, window_size=1, algo='viterbi'):
    test_reader = ICHISeqDataReader(test_names)
    test_set_x, test_set_y = test_reader.read_all()
        
    for test_patient in test_names:
        #get data divided on sequences with respect to labels
        test_set_x, test_set_y = test_reader.read_next_doc()
        test_set_x = test_set_x.get_value()
        test_set_y = test_set_y.eval()
        n_test_times = test_set_x.shape[0] - window_size
        
        test_visible_after_sda = numpy.array([sda.get_da_output(
                test_set_x[time: time+window_size]).ravel()
                for time in xrange(n_test_times)])
                    
        new_test_visible = create_labels_after_das(
            da_output_matrix=test_visible_after_sda,
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        
        n_patient_samples = len(test_set_y)
        half_window_size = int(window_size/2)
        new_test_hidden=test_set_y[half_window_size:n_patient_samples-half_window_size]
        
        patient_error = get_error_on_patient2(
            model=sda.hmmLayer,
            visible_set=new_test_visible,
            hidden_set=new_test_hidden,
            algo=algo,
            pat = test_patient
        )
        
        print(patient_error, ' error for patient ' + test_patient)
        gc.collect()  
    
def test_all_params():
    window_size = 100
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    train_data = ['p002', 'p003', 'p005', 'p08b']
    test_data = all_train
    
    corruption_levels = [.1, .2]
    pretrain_lr=.03
    
    rank = 3
    
    output_folder=('%s, all_test')%(train_data)
    trained_sda = train_SdA(
        train_names=train_data,
        output_folder=output_folder,
        base_folder='SdA_second_HMM',
        window_size=window_size,
        corruption_levels=corruption_levels,
        pretrain_lr=pretrain_lr,
        rank=rank,
        pretraining_epochs=15
    )
    test_sda(sda=trained_sda,
            test_names=test_data,
            rank=rank,
            window_size=window_size
        )
    '''
    train_data = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
                  'p10a','p011','p012','p013','p014','p15a','p15b','p016',
                  'p017','p018','p019','p020','p021','p022','p023','p025',
                  'p026','p027','p028','p029','p030','p031','p032','p033',
                  'p034','p035','p036','p037','p038','p040','p042','p043',
                  'p044','p045','p047','p048','p049','p050','p051']
    for test_pat_num in xrange(len(train_data)):
        test_pat = train_data.pop(test_pat_num)
        output_folder=('all_data, [%s]')%(test_pat)
        trained_sda = train_SdA(
            train_names=train_data,
            output_folder=output_folder,
            base_folder='SdA_second_HMM',
            window_size=window_size,
            corruption_levels=corruption_levels,
            pretrain_lr=pretrain_lr,
            start_base=start_base,
            rank=rank,
            pretraining_epochs=15
        )
        test_sda(sda=trained_sda,
            test_names=[test_pat],
            rank=rank,
            start_base=start_base,
            window_size=window_size
        )
        train_data.insert(test_pat_num, test_pat)
    '''

def create_labels_after_das(da_output_matrix, rank, window_size, start_base=10):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    mins = da_output_matrix.min(axis=0)
    maxs = da_output_matrix.max(axis=0)
    da_output_matrix = ((da_output_matrix-mins)*((1-(-1.))/(maxs-mins)))/2
    #get average and dispersion
    avg_disp_matrix = numpy.array([[da_output_matrix[i].mean(),
                         da_output_matrix[i].max()-
                         da_output_matrix[i].min()]
        for i in xrange(da_output_matrix.shape[0])])
    base = pow(start_base, rank) + 1
    arounded_matrix = numpy.rint(avg_disp_matrix.flatten()*pow(start_base, rank)).reshape((da_output_matrix.shape[0], 2))
    data_labels = []
    #n_in=2
    for row in arounded_matrix:
        data_labels.append(int(row[0]*base + row[1]))
    return data_labels

if __name__ == '__main__':
    test_all_params()