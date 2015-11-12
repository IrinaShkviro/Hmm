# -*- coding: utf-8 -*-
"""
Created on Sun Nov 08 11:00:45 2015

@author: irka
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:02:02 2015

@author: Ren
"""

import gc

import numpy

import theano
import theano.tensor as T

from ichi_reader import ICHISeqDataReader
        

def errors(actual_states):
    """Return 1 if y!=y_predicted (error) and 0 if right

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives for each example the
                  correct label
    """
    # check if y has same dimension of y_pred        
    predict = numpy.array([0 for i in xrange(actual_states.shape[0])])
                
    actual = numpy.array([1
        if actual_states[i] ==4
        else 0
        for i in xrange(actual_states.shape[0])])

    # the T.neq operator returns a vector of 0s and 1s, where 1
    # represents a mistake in prediction
    return T.neq(theano.shared(predict), theano.shared(actual))

def get_error_on_patient(hidden_set):
    error_array=errors(actual_states=hidden_set)
    return error_array.eval().mean()
                        
def train_separately():
    all_train = ['p002','p003','p005','p007','p08a','p08b','p09a','p09b',
			'p10a','p011','p012','p013','p014','p15a','p15b','p016',
               'p017','p018','p019','p020','p021','p022','p023','p025',
               'p026','p027','p028','p029','p030','p031','p032','p033',
               'p034','p035','p036','p037','p038','p040','p042','p043',
               'p044','p045','p047','p048','p049','p050','p051']
    valid_data = all_train
        
    valid_reader = ICHISeqDataReader(valid_data)
    
    for valid_patient in valid_data:
        #get data divided on sequences with respect to labels
        valid_set_x, valid_set_y = valid_reader.read_next_doc()
        
        patient_error = get_error_on_patient(
            hidden_set=valid_set_y.eval()
        )
        
        print(patient_error, ' error for patient ' + valid_patient)

        gc.collect()  
                   
if __name__ == '__main__':
    train_separately()