# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 23:51:34 2015

@author: irka
"""

import gc

import numpy

from sklearn import hmm

from base import errors, errors2, confusion_matrix
from ichi_reader import ICHISeqDataReader

def update_params_on_patient(pi_values, a_values, b_values, array_from_hidden,
                             hiddens_patient, visibles_patient, n_hidden):
    pi_values[hiddens_patient[0]] += 1
    n_patient_samples = len(hiddens_patient)
    for index in xrange(n_patient_samples-1):
        a_values[hiddens_patient[index], hiddens_patient[index+1]] += 1
        b_values[hiddens_patient[index], visibles_patient[index]] += 1            
    b_values[hiddens_patient[n_patient_samples-1],
             visibles_patient[n_patient_samples-1]] += 1
    for hidden in xrange(n_hidden):
        array_from_hidden[hidden] += len(hiddens_patient[numpy.where(hiddens_patient==hidden)])
    array_from_hidden[hiddens_patient[n_patient_samples-1]] -= 1
    return (pi_values, a_values, b_values, array_from_hidden)

def finish_training(pi_values, a_values, b_values, array_from_hidden, n_hidden,
                    n_patients):
    for hidden in xrange(n_hidden):
        a_values[hidden] = a_values[hidden]/array_from_hidden[hidden]
        b_values[hidden] = b_values[hidden]/array_from_hidden[hidden]
    pi_values = pi_values/float(n_patients)
    return (pi_values, a_values, b_values)

def get_error_on_patient(model, visible_set, hidden_set, algo, all_labels = True, pat = ""):
    predicted_states = model.predict(
            obs=visible_set,
            algorithm=algo
    )
    if all_labels:
        error_array=errors(
            predicted=predicted_states,
            actual=hidden_set
        )
        return error_array.eval().mean()
    else:
        error_array = confusion_matrix(
            predicted_states = predicted_states,
            actual_states = hidden_set,
            pat = pat
        )
        
        return error_array
                        
def train():
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    train_data_names = all_train
    valid_data = ['p012']
    
    '''
    train_mask = numpy.in1d(all_train, train_data_names)
    train_indexes = numpy.where(train_mask)
    valid_data = numpy.delete(valid_data, train_indexes)
    '''

    n_train_patients=len(train_data_names)
    
    rank = 5
    window_size = 20
    n_visible=pow(10, rank) + 2 - window_size
    n_hidden=7
    preprocess_algo = "filter+avg"
    if (preprocess_algo == "avg_disp" or preprocess_algo == "filter+avg_disp"):
        n_visible *= 10
        
    train_reader = ICHISeqDataReader(train_data_names)
    valid_reader = ICHISeqDataReader(valid_data)
    
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_next_doc(
            algo = preprocess_algo,
            rank = rank,
            window = window_size
        )
        
        pi_values, a_values, b_values, array_from_hidden = update_params_on_patient(
            pi_values=pi_values,
            a_values=a_values,
            b_values=b_values,
            array_from_hidden=array_from_hidden,
            hiddens_patient=train_set_y.eval(),
            visibles_patient=train_set_x.eval(),
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
    
    #use standart model of hmm
    hmm_model = hmm.MultinomialHMM(
        n_components = n_hidden,
        startprob = pi_values,
        transmat = a_values
    )
    hmm_model.n_symbols = n_visible
    hmm_model.emissionprob_ = b_values 
    gc.collect()
    print('MultinomialHMM created')
    predict_algo = 'viterbi'
    
    error_array = []

    for valid_patient in valid_data:
        #get data divided on sequences with respect to labels
        valid_set_x, valid_set_y = valid_reader.read_next_doc(
            algo = preprocess_algo,
            rank = rank,
            window = window_size
        )
        
        patient_error = get_error_on_patient(
            model=hmm_model,
            visible_set=valid_set_x.eval(),
            hidden_set=valid_set_y.eval(),
            algo=predict_algo,
            pat = valid_patient,
            all_labels = False
        )
        
        #print(patient_error, ' error for patient ' + valid_patient)
        error_array.append(patient_error)
        gc.collect() 
    '''
    print(error_array)             
    print('mean value of error: ', numpy.round(numpy.mean(error_array), 6))
    print('min value of error: ', numpy.round(numpy.amin(error_array), 6))
    print('max value of error: ', numpy.round(numpy.amax(error_array), 6))
    '''
    
    total_conf_matrix = numpy.sum(error_array, axis = 0)
    
    print('total true_sleep: ', total_conf_matrix[0])
    print('total false_wake: ', total_conf_matrix[1])
    print('total false_sleep: ', total_conf_matrix[2])
    print('total true_wake: ', total_conf_matrix[3])

if __name__ == '__main__':
    train()