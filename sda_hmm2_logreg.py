# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 13:01:26 2016

@author: Serg
"""

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
from hmmlearn import hmm

import theano
import theano.tensor as T

from hmm1 import GeneralHMM, mean_error
from hmm2 import update_params_on_patient, finish_training, get_error_on_patient
from sda import pretrain_SdA
from ichi_reader import ICHISeqDataReader
#from cg import pretrain_sda_cg, finetune_sda_cg
from sgd import finetune_log_layer_sgd
from preprocess import filter_data, create_int_labels, create_av_disp, create_av

theano.config.exception_verbosity='high'

def finetune_hmm2(sda,
                  read_window,
                  read_algo,
                  read_rank,
                  posttrain_rank,
                  posttrain_algo,
                  window_size,
                  train_names):
                     
    n_train_patients=len(train_names)
    
    n_visible = pow(10, posttrain_rank) + 2 - read_window #input of sda
    n_visible = n_visible - window_size + 1 #output of sda
    n_hidden = 7
    
    posttrain_window = sda.da_layers_output_size
        
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
            [sda.get_da_output(
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
        n_components=n_hidden
    )
    
    hmm_model.startprob_ = pi_values
    hmm_model.transmat_ = a_values
    hmm_model.n_symbols = n_visible
    hmm_model.emissionprob_ = b_values 
    gc.collect()
    print('MultinomialHMM created')
    
    sda.set_hmm2(
        hmm2 = hmm_model
    )
    
    return sda
    
def validate_model(sda,
                   valid_names,
                   read_window,
                   read_algo,
                   read_rank,
                   window_size):
                       
    valid_reader = ICHISeqDataReader(valid_names)
    valid_errors = []
    for i in xrange (len(valid_names)):
        valid_x, valid_y = valid_reader.read_next_doc(
            algo = read_algo,
            rank = read_rank,
            window = read_window,
            divide = False
        )
        valid_x = valid_x.get_value()
        valid_y = valid_y.eval()
        
        n_valid_times = valid_x.shape[0] - window_size + 1
                    
        new_valid_x = numpy.array(
            [sda.get_da_output(
                    valid_x[time: time + window_size]
                ).ravel()
            for time in xrange(n_valid_times)]
        )

        half_window_size = int(window_size/2)
        new_valid_y = valid_y[
            half_window_size: n_valid_times + half_window_size
        ]

        #compute mean error value for patients in validation set
        pat_error = mean_error(
            gen_hmm = sda.hmm1,
            obs_seq = new_valid_x,
            actual_states = new_valid_y
        )
        valid_errors.append(pat_error)
    return numpy.mean(valid_errors)
    
def finetune_hmm1(sda,
                  n_components,
                  n_hmms,
                  train_names,
                  valid_names,
                  global_epochs,
                  read_rank,
                  read_window,
                  read_algo,
                  window_size,
                  posttrain_algo,
                  posttrain_rank,
                  posttrain_window):
                      
    # set hmm1 layer on sda
    sda.set_hmm1(
        hmm1 = GeneralHMM(
            n_components = n_components,
            n_hmms = n_hmms
        )        
    )
    
    for epoch in xrange(global_epochs):
        train_reader = ICHISeqDataReader(train_names)
        n_train_patients = len(train_names)
        
        #train hmms on data of each patient
        for train_patient in xrange(n_train_patients):
            #get data divided on sequences with respect to labels
            train_set = train_reader.read_next_doc(
                algo = read_algo,
                rank = read_rank,
                window = read_window,
                divide = True
            )
            for label in xrange(n_hmms):
                train_for_label = train_set[label].eval()
                if train_for_label != []:
                    n_train_times = train_for_label.shape[0] - window_size + 1
                    
                    train_after_sda = numpy.array(
                        [sda.get_da_output(
                            train_for_label[time: time + window_size]
                        ).ravel()
                        for time in xrange(n_train_times)]
                    )
                    
                    if train_after_sda.shape[0] > sda.hmm1.hmm_models[label].n_components:
                        sda.hmm1.hmm_models[label].fit(
                            train_after_sda.reshape((-1, 1))
                        )
                        sda.hmm1.isFitted[label] = True
                            
            error_cur_epoch = validate_model(
                sda = sda,
                valid_names = valid_names,
                read_window = read_window,
                read_algo = read_algo,
                read_rank = read_rank,
                window_size = window_size
            )
            sda.hmm1.valid_error_array.append([])
            sda.hmm1.valid_error_array[-1].append(
                epoch*n_train_patients + train_patient
            )
            sda.hmm1.valid_error_array[-1].append(error_cur_epoch)
                
            gc.collect()
            
    return sda

def train_SdA(train_names,
              valid_names,
              read_window,
              read_algo,
              read_rank,
              window_size,
              corruption_levels,
              hidden_layers_sizes,
              pretraining_epochs,
              pretraining_pat_epochs,
              pretrain_lr,
              pretrain_algo,
              output_folder,
              base_folder,              
              posttrain_rank,
              posttrain_algo,
              posttrain_window,
              finetune_algo,
              finetune_lr,
              global_epochs,
              pat_epochs):
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
        valid_names = valid_names,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        window_size = window_size,
        corruption_levels = corruption_levels,
        pretraining_epochs = pretraining_epochs,
        pretraining_pat_epochs = pretraining_pat_epochs,
        pretrain_lr = pretraining_epochs,        
        pretrain_algo = pretrain_algo,
        hidden_layers_sizes = hidden_layers_sizes,
        output_folder = output_folder,
        base_folder = base_folder       
    )
    
    end_time = timeit.default_timer()
        
    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # train hmm1 layer
    hmms_count = 7
    n_hiddens = [5]*hmms_count
    
    finetuned_sda_hmm1 = finetune_hmm1(
        sda = pretrained_sda,
        n_components = n_hiddens,
        n_hmms = hmms_count,
        train_names = train_names,
        valid_names = valid_names,
        global_epochs = global_epochs,
        read_rank = read_rank,
        read_window = read_window,
        read_algo = read_algo,
        window_size = window_size,
        posttrain_algo = posttrain_algo,
        posttrain_rank = posttrain_rank,
        posttrain_window = posttrain_window
    )
    
    # calculate hmm2 layer
    finetuned_sda_hmm2 = finetune_hmm2(
        sda = finetuned_sda_hmm1,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        posttrain_rank = posttrain_rank,
        posttrain_algo = posttrain_algo,
        window_size = window_size,
        train_names = train_names
    )
    
    # train logistic regression layer
    if finetune_algo == 'sgd':    
        finetuned_sda = finetune_log_layer_sgd(
            sda = finetuned_sda_hmm2,
            train_names = train_names,
            valid_names = valid_names,
            read_algo = read_algo,
            read_window = read_window,
            read_rank = read_rank,
            window_size = window_size,
            finetune_lr = finetune_lr,
            global_epochs = global_epochs,
            pat_epochs = pat_epochs,
            output_folder = output_folder
        )
    else:        
        finetuned_sda = finetuned_sda_hmm2
    


    return finetuned_sda
    
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
    
    index = T.lscalar('index')
    test_set_x = T.vector('test_set_x')
    test_set_y = T.ivector('test_set_y')
    y = T.iscalar('y')  # labels, presented as int label
    
    hmm1_error_array = []
    hmm2_error_array = []
    log_reg_errors = []
    
    test_log_reg = theano.function(
        inputs=[
            index,
            test_set_x,
            test_set_y
        ],
        outputs=[sda.logLayer.errors(y), sda.logLayer.predict(), y],
        givens={
            sda.x: test_set_x[index: index + window_size],
            y: test_set_y[index + window_size - 1]
        }
    )    
    
    for test_patient in test_names:
        test_set_x, test_set_y = test_reader.read_next_doc(
            algo = read_algo,
            window = read_window,
            rank = read_rank
        )
                        
        test_set_x = test_set_x.get_value(borrow=True)
        test_set_y = test_set_y.eval()
        
        n_test_times = test_set_x.shape[0] - window_size + 1
        
        test_result = [test_log_reg(
            index = i,
            test_set_x = test_set_x,
            test_set_y = test_set_y) for i in xrange(n_test_times)
        ]
        test_result = numpy.asarray(test_result)
        test_losses = test_result[:,0]
        test_score = float(numpy.mean(test_losses))*100
                            
        log_reg_errors.append(test_score)
        
        
        test_visible_after_sda = numpy.array(
            [sda.get_da_output(
                test_set_x[time: time+window_size]
            ).ravel()
            for time in xrange(n_test_times)]
        )
        
        half_window_size = int(window_size/2)
        test_y_after_sda = test_set_y[
            half_window_size : n_test_times + half_window_size
        ]
        
        pat_error = mean_error(
            gen_hmm = sda.hmm1,
            obs_seq = test_visible_after_sda,
            actual_states = test_y_after_sda
        )
        hmm1_error_array.append(pat_error)
                            
        new_test_visible = create_labels_after_das(
            da_output_matrix = test_visible_after_sda,
            algo = posttrain_algo,
            rank = posttrain_rank,
            window = posttrain_window
        )
        
        n_patient_samples = len(new_test_visible)
        new_test_hidden = test_set_y[half_window_size:n_patient_samples+half_window_size]
        
        patient_error = get_error_on_patient(
            model = sda.hmm2,
            visible_set = new_test_visible,
            hidden_set = new_test_hidden,
            algo = predict_algo,
            pat = test_patient,
            all_labels = True
        )
        
        hmm2_error_array.append(patient_error)
        print(pat_error, ' error (hmm1) for patient ' + test_patient)
        print(patient_error, ' error (hmm2) for patient ' + test_patient)
        print(test_score, ' error (log_reg) for patient ' + test_patient)
        gc.collect()
        
    return hmm1_error_array, hmm2_error_array, log_reg_errors
    
def test_all_params():
    
    window_size = 30
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    without_valid = ['p002','p003','p08a','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    valid_names = ['p007', 'p028', 'p005', 'p08b']
    test_data = ['p002', 'p017', 'p014', 'p048']
    
    read_window = 20
    read_algo = "filter+avg"
    read_rank = 10
    
    corruption_levels = [.1, .2]
    hidden_layers_sizes = [window_size/2, window_size/3]
    pretraining_epochs = 1
    pretraining_pat_epochs = 1
    pretrain_lr=.03        
    pretrain_algo = "sgd"
    
    posttrain_rank = 5    
    posttrain_algo = "avg"
    posttrain_window = 1
    
    finetune_algo = 'sgd'
    finetune_lr = 0.003
    
    global_epochs = 1
    pat_epochs = 1
    
    output_folder=('all_train, %s')%(test_data)

    trained_sda = train_SdA(    
        train_names = ['p002'],
        valid_names = valid_names,
        read_window = read_window,
        read_algo = read_algo,
        read_rank = read_rank,
        window_size = window_size,
        corruption_levels = corruption_levels,
        hidden_layers_sizes = hidden_layers_sizes,
        pretraining_epochs = pretraining_epochs,
        pretrain_lr = pretrain_lr,
        pretrain_algo = pretrain_algo,
        pretraining_pat_epochs = pretraining_pat_epochs,
        output_folder = output_folder,
        base_folder = 'sda_hmm2', 
        posttrain_rank = posttrain_rank,
        posttrain_algo = posttrain_algo,
        posttrain_window = posttrain_window,
        finetune_algo = finetune_algo,
        finetune_lr = finetune_lr,
        global_epochs = global_epochs,
        pat_epochs = pat_epochs
    )
    
    hmm1_errors, hmm2_errors, log_reg_errors = test_sda(
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
    
    print(hmm1_errors, 'hmm1')
    print('mean hmm value of error: ', numpy.round(numpy.mean(hmm1_errors), 6))
    print('min hmm value of error: ', numpy.round(numpy.amin(hmm1_errors), 6))
    print('max hmm value of error: ', numpy.round(numpy.amax(hmm1_errors), 6))
        
    print(hmm2_errors, 'hmm2')
    print('mean hmm value of error: ', numpy.round(numpy.mean(hmm2_errors), 6))
    print('min hmm value of error: ', numpy.round(numpy.amin(hmm2_errors), 6))
    print('max hmm value of error: ', numpy.round(numpy.amax(hmm2_errors), 6))
    
    print(log_reg_errors, 'log_reg')
    print('mean reg value of error: ', numpy.round(numpy.mean(log_reg_errors), 6))
    print('min reg value of error: ', numpy.round(numpy.amin(log_reg_errors), 6))
    print('max reg value of error: ', numpy.round(numpy.amax(log_reg_errors), 6))
    
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
    print sys.stdout.encoding
    test_all_params()
