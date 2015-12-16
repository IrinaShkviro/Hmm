# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 23:51:34 2015

@author: irka
"""

import gc
import numpy
import xlrd, xlwt

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
 
def create_hmm(
    train_data_names,
    n_hidden,
    n_visible,
    read_algo,
    read_rank,
    read_window):
        
    train_reader = ICHISeqDataReader(train_data_names)
    
    n_train_patients=len(train_data_names)
    
    pi_values = numpy.zeros((n_hidden,))
    a_values = numpy.zeros((n_hidden, n_hidden)) 
    b_values = numpy.zeros((n_hidden, n_visible))
    array_from_hidden = numpy.zeros((n_hidden,))

    for train_patient in xrange(n_train_patients):
        #get data divided on sequences with respect to labels
        train_set_x, train_set_y = train_reader.read_next_doc(
            algo = read_algo,
            rank = read_rank,
            window = read_window
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
    
    return hmm_model
    
def test(
    hmm_model,
    valid_data,
    read_algo,
    read_window,
    read_rank,
    predict_algo):
        
    valid_reader = ICHISeqDataReader(valid_data)

    for valid_patient in valid_data:
        #get data divided on sequences with respect to labels
        valid_set_x, valid_set_y = valid_reader.read_next_doc(
            algo = read_algo,
            rank = read_rank,
            window = read_window
        )
        
        patient_error = get_error_on_patient(
            model=hmm_model,
            visible_set=valid_set_x.eval(),
            hidden_set=valid_set_y.eval(),
            algo=predict_algo,
            pat = valid_patient,
            all_labels = True
        )
        
        print(patient_error, ' error for patient ' + valid_patient)
        gc.collect()
    return patient_error
                           
def train():
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    train_data = all_train
    
    '''
    train_mask = numpy.in1d(all_train, train_data_names)
    train_indexes = numpy.where(train_mask)
    valid_data = numpy.delete(valid_data, train_indexes)
    '''
    
    read_rank = 5
    read_window = 20
    n_visible=pow(10, read_rank) + 2 - read_window
    n_hidden=7
    read_algo = "filter+avg"
    if (read_algo == "avg_disp" or read_algo == "filter+avg_disp"):
        n_visible *= 10
       
    predict_algo = 'viterbi'
    
    result_list = xlwt.Workbook()
    result_sheet = result_list.add_sheet('Test')
    for test_pat_num in xrange(len(train_data)):
        test_pat = train_data.pop(test_pat_num)
        #output_folder=('all_data, [%s]')%(test_pat)
        hmm_model = create_hmm(
            train_data_names = train_data,
            n_hidden = n_hidden,
            n_visible = n_visible,
            read_algo = read_algo,
            read_rank = read_rank,
            read_window = read_window
        )
        
        error = test(
            hmm_model = hmm_model,
            valid_data = [test_pat],
            read_algo = read_algo,
            read_window = read_window,
            read_rank = read_rank,
            predict_algo = predict_algo
        )
        result_sheet.write(test_pat_num, 0, test_pat)
        result_sheet.write(test_pat_num, 1, error)
        train_data.insert(test_pat_num, test_pat)   
    result_list.save('hmm2.xls')

if __name__ == '__main__':
    train()