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

from hmm2 import update_params_on_patient, finish_training, get_error_on_patient
from sda import pretrain_SdA
#from MyVisualizer import visualize_pretraining, visualize_finetuning
from ichi_reader import ICHISeqDataReader
from cg import pretrain_sda_cg
from sgd import pretrain_sda_sgd
from preprocess import filter_data, create_int_labels, create_av_disp, create_av

theano.config.exception_verbosity='high'

def train_SdA(train_names,
              read_window,
              read_algo,
              read_rank,
              window_size,
              corruption_levels,
              hidden_layers_sizes,
              pretraining_epochs,
              pretrain_lr,
              pretrain_algo,
              output_folder,
              base_folder,              
              posttrain_rank,
              posttrain_algo):
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
 
    start_time = timeit.default_timer()
       
    pretrained_sda = pretrain_SdA(
        train_names = train_names,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        window_size = window_size,
        corruption_levels = corruption_levels,
        pretraining_epochs = pretraining_epochs,
        pretrain_lr = pretraining_epochs,        
        pretrain_algo = pretrain_algo,
        hidden_layers_sizes = hidden_layers_sizes,
        output_folder = output_folder,
        base_folder = base_folder       
    )
    
    end_time = timeit.default_timer()
        
    '''
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
    '''
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    n_train_patients=len(train_names)
    
    n_visible = pow(10, posttrain_rank) + 2 - read_window #input of sda
    n_visible = n_visible - window_size + 1 #output of sda
    n_hidden = 7
    
    posttrain_window = pretrained_sda.da_layers_output_size
        
    train_reader = ICHISeqDataReader(train_names)
    
    #create matrices for params of HMM layer
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden))
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))
    
    if (posttrain_algo == "avg_disp" or posttrain_algo == "filter+avg_disp"):
        n_visible *= 10
        
    for train_patient in xrange(n_train_patients):
        train_set_x, train_set_y = train_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
        train_set_x = train_set_x.get_value()
        train_set_y = train_set_y.eval()
        
        n_train_times = train_set_x.shape[0] - window_size + 1
        
        train_visible_after_sda = numpy.array(
            [pretrained_sda.get_da_output(
                train_set_x[time: time + window_size]
            ).ravel()
            for time in xrange(n_train_times)]
        )
             
        new_train_visible = create_labels_after_das(
            da_output_matrix = train_visible_after_sda,
            algo = posttrain_algo,
            rank = posttrain_rank,
            window = posttrain_window
        )
        n_patient_samples = len(new_train_visible)
        half_window_size = int(window_size/2)
        new_train_hidden=train_set_y[half_window_size:n_patient_samples+half_window_size]
        
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
    
    pretrained_sda.set_classifier(
        classifier = hmm_model
    )
    return pretrained_sda
    
def test_sda(
    sda,
    test_names,
    read_window,
    read_algo,
    read_rank,
    window_size,
    posttrain_rank,
    posttrain_algo,
    predict_algo='viterbi'):

    test_reader = ICHISeqDataReader(test_names)
    posttrain_window = sda.da_layers_output_size
    
    error_array = []
    for test_patient in test_names:
        test_set_x, test_set_y = test_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
        test_set_x = test_set_x.get_value()
        test_set_y = test_set_y.eval()
        n_test_times = test_set_x.shape[0] - window_size + 1
        
        test_visible_after_sda = numpy.array(
            [sda.get_da_output(
                test_set_x[time: time+window_size]
            ).ravel()
            for time in xrange(n_test_times)]
        )
                    
        new_test_visible = create_labels_after_das(
            da_output_matrix = test_visible_after_sda,
            algo = posttrain_algo,
            rank = posttrain_rank,
            window = posttrain_window
        )
        
        n_patient_samples = len(new_test_visible)
        half_window_size = int(window_size/2)
        new_test_hidden = test_set_y[half_window_size:n_patient_samples+half_window_size]
        
        patient_error = get_error_on_patient(
            model = sda.classifier,
            visible_set = new_test_visible,
            hidden_set = new_test_hidden,
            algo = predict_algo,
            pat = test_patient,
            all_labels = True
        )
        
        error_array.append(patient_error)
        print(patient_error, ' error for patient ' + test_patient)
        gc.collect()
        
    return error_array
    
def test_all_params():
    window_size = 100
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    without_test = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p012','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    train_data = without_test
    test_data = ['p013', 'p038']
    
    read_window = 1
    read_algo = "filter"
    read_rank = 20
    
    corruption_levels = [.1, .2]
    hidden_layers_sizes = [window_size/2, window_size/3]
    pretraining_epochs = 15
    pretrain_lr=.03        
    pretrain_algo = "sgd"
    
    posttrain_rank = 5    
    posttrain_algo = "avg"
    
    output_folder=('all_train, %s')%(test_data)

    trained_sda = train_SdA(    
        train_names = train_data,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        window_size = window_size,
        corruption_levels = corruption_levels,
        hidden_layers_sizes = hidden_layers_sizes,
        pretraining_epochs = pretraining_epochs,
        pretrain_lr = pretrain_lr,
        pretrain_algo = pretrain_algo,
        output_folder = output_folder,
        base_folder = 'sda_hmm2', 
        posttrain_rank = posttrain_rank,
        posttrain_algo = posttrain_algo              
    )
    
    error_array = test_sda(
        sda = trained_sda,
        test_names = test_data,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        window_size = window_size,
        posttrain_rank = posttrain_rank,
        posttrain_algo = posttrain_algo,
        predict_algo = 'viterbi'
    )
    
    print(error_array)             
    print('mean value of error: ', numpy.round(numpy.mean(error_array), 6))
    print('min value of error: ', numpy.round(numpy.amin(error_array), 6))
    print('max value of error: ', numpy.round(numpy.amax(error_array), 6))
    
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

def create_labels_after_das(da_output_matrix, algo, rank, window):
    return [create_label_after_das(da_output_matrix[i], algo, rank, window)
        for i in xrange(len(da_output_matrix))]

def create_label_after_das(da_output_seq, algo, rank, window):
    if algo == "avg_disp":
        d_x = create_av_disp(
            sequence = da_output_seq,
            rank = rank,
            window_size = window
        )
    elif algo == "avg":
        d_x = create_av(
            sequence = da_output_seq,
            rank = rank,
            window_size = window
        )
    elif algo == "filter+avg":
        d_x = create_av(
            sequence = filter_data(da_output_seq),
            rank = rank,
            window_size = window
        )
    elif algo == "filter+avg_disp":
        d_x = create_av_disp(
            sequence = filter_data(da_output_seq),
            rank = rank,
            window_size = window
        )
    else:
        d_x = [numpy.mean(create_int_labels(
            sequence = da_output_seq,
            rank = rank
        ))]
    return d_x[0]

if __name__ == '__main__':
    test_all_params()