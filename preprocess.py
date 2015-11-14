# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 20:27:07 2015

@author: irka
"""

import numpy
import scipy.signal as signal
import matplotlib.pyplot as plt

def normalize_sequence_1_1(sequence):
    """
    Normalize sequence
    Return sequence with values from -1 to 1
    """
    mins = numpy.amin(sequence)
    maxs = numpy.amax(sequence)
    sequence = (sequence-mins)*((1-(-1.))/(maxs-mins)) - 1
    return sequence

def normalize_sequence_0_1(sequence):
    """
    Normalize sequence
    Return sequence with values from 0 to 1
    """
    mins = numpy.amin(sequence)
    maxs = numpy.amax(sequence)
    sequence = ((sequence-mins)*((1-(-1.))/(maxs-mins)))/2
    return sequence
    
def filter_data(sequence):
    #filter the data with a butterworth filter
    
    # First, design the Buterworth filter
    N  = 3   # Filter order
    Wn = 0.25 # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    
    sequence = normalize_sequence_1_1(sequence)
    
    # Second, apply the filter
    sequencef = signal.filtfilt(B,A, sequence)
    
    '''
    # Make plots
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    plt.plot(sequence, 'b-')
    plt.plot(sequencef, 'r-',linewidth=2)
    plt.ylabel("z-coord")
    plt.legend(['Original','Filtered'])
    ax1.axes.get_xaxis().set_visible(False)
    '''
    return sequencef
    
def create_int_labels(sequence, rank):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    sequence = normalize_sequence_0_1(sequence)
    int_matrix = (numpy.around(sequence, rank)*pow(10, rank)).astype(int)
    return int_matrix

def create_av_disp(sequence, rank, window_size=1):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    sequence = normalize_sequence_0_1(sequence)
    #get average and dispersion
    avg_disp_matrix = [[numpy.mean(sequence[i: i + window_size]),
                        numpy.amax(sequence[i: i + window_size])-
                        numpy.amin(sequence[i: i + window_size])]
        for i in xrange(len(sequence)-window_size+1)]
    #11^3 variant for labels
    arounded_matrix = numpy.around(avg_disp_matrix, rank)*pow(10, rank)
    data_labels = []
    for row in arounded_matrix:
        new_row = row.flat
        #create individual labels for vectors
        cur_value = new_row[0] * 10 + new_row[1]
        data_labels.append(int(cur_value))
    data_labels = data_labels - numpy.amin(data_labels)
    return data_labels
    
def create_av(sequence, rank=-1, window_size=1):
    """
    Normalize sequence matrix and get average and dispersion
    """
    #normalization
    sequence = normalize_sequence_0_1(sequence)
    #get average and dispersion
    avg_seq = [numpy.mean(sequence[i: i + window_size])
        for i in xrange(len(sequence)-window_size+1)]
    if rank > -1:
        arounded_seq = numpy.around(avg_seq, rank)*pow(10, rank)
        arounded_seq = arounded_seq.astype(int)
        result_seq = arounded_seq - numpy.amin(arounded_seq)
    else:
        result_seq = avg_seq - numpy.amin(avg_seq)
    return result_seq    
    
if __name__ == '__main__':
    array = [3, 3,3,3,50]
    array2 = [1]*len(array)
    res = numpy.array(array) + 1
    print(res)