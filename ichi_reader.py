# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 20:16:45 2015

@author: irka
"""

import numpy
import gc

import theano
import theano.tensor as T

from preprocess import filter_data, normalize_sequence_1_1,\
    normalize_sequence_0_1, create_int_labels, create_av_disp, create_av

class ICHISeqDataReader(object):
    def __init__(self, seqs_for_analyse):
        #print "init ICHISeqDataReader"
        
        #seqs - files for each patient
        self.seqs = seqs_for_analyse
        
        #n - count of patients
        self.n = len(self.seqs)
        
        #n_in - count of marks (dimension of input data)
        self.n_in = 1
        
        self.sequence_index = 0
        # path to folder with data
        dataset = 'D:\Irka\Projects\NeuralNetwok\data\data' # "./data/7/ICHI14_data_set/data"
        self.init_sequence(dataset)
    
    # read all docs in sequence
    def read_all(self):
        # sequence_matrix = array[size of 1st doc][ data.z, data.gt]
        sequence_matrix = self.get_sequence()

        # d_x1 = array[size of 1st doc][z]
        d_x1 = filter_data(
            sequence = sequence_matrix[:, self.n_in-1]
        )
        
        # d_y1 = array[size of 1st doc][labels]
        d_y1 = sequence_matrix[:, self.n_in]

        # data_x_ar = union for z-coordinate in all files
        data_x = d_x1
        
        # data_y_ar = union for labels in all files
        data_y = d_y1
        
        for t in range(len(self.seqs) - 1):
            # sequence_matrix = array[size of t-th doc][data.z, data.gt]
            sequence_matrix = self.get_sequence()

            # d_x = array[size of t-th doc][z]
            d_x = filter_data(
                sequence = sequence_matrix[:, self.n_in-1]
            )
            
            # d_y = array[size of t-th doc][labels]
            d_y = sequence_matrix[:, self.n_in]
            
            # concatenate data in current file with data in prev files in one array
            data_x = numpy.concatenate((data_x, d_x))
            data_y = numpy.concatenate((data_y, d_y))
                            
            gc.collect()
        
        set_x = theano.shared(numpy.asarray(data_x,
                                                   dtype=theano.config.floatX),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(data_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        
        return (set_x, set_y) 
    
    # read one doc in sequence
    def read_next_doc(self, algo, rank=1, window=1):    
       
        # sequence_matrix = array[size of doc][data.z, data.gt]
        sequence_matrix = self.get_sequence()
        
        # d_x = array[size of doc][z]
        if algo == "filter":
            d_x = filter_data(
                sequence = sequence_matrix[:, self.n_in-1]
            )
        elif algo == "normalize_1_1":
            d_x = normalize_sequence_1_1(
                sequence = sequence_matrix[:, self.n_in-1]
            )
        elif algo == "normalize_0_1":
            d_x = normalize_sequence_0_1(
                sequence = sequence_matrix[:, self.n_in-1]
            )
        elif algo == "int_labels":
            d_x = create_int_labels(
                sequence = sequence_matrix[:, self.n_in-1],
                rank = rank
            )
        elif algo == "avg_disp":
            d_x = create_av_disp(
                sequence = sequence_matrix[:, self.n_in-1],
                rank = rank,
                window_size = window
            )
        elif algo == "avg":
            d_x = create_av(
                sequence = sequence_matrix[:, self.n_in-1],
                rank = rank,
                window_size = window
            )
        elif algo == "filter+avg":
            d_x = create_av(
                sequence = filter_data(sequence_matrix[:, self.n_in-1]),
                rank = rank,
                window_size = window
            )
        elif algo == "filter+avg_disp":
            d_x = create_av_disp(
                sequence = filter_data(sequence_matrix[:, self.n_in-1]),
                rank = rank,
                window_size = window
            )
        else:
            d_x = sequence_matrix[:, self.n_in-1]
        
        # d_y = array[size of doc][labels]
        d_y = sequence_matrix[:, self.n_in]
        d_y = d_y[window/2: len(d_y) + window/2 -window +1]
           
        gc.collect()
        
        set_x = theano.shared(numpy.asarray(d_x),
                                     borrow=True)
        set_y = T.cast(theano.shared(numpy.asarray(d_y,
                                                   dtype=theano.config.floatX),
                                     borrow=True), 'int32')
        
        return (set_x, set_y)
        
    def init_sequence(self, dataset):
        self.sequence_files = []
        
        for f in self.seqs:
            # sequence_file - full path to each document
            sequence_file = dataset+"/"+str(f)+".npy"
            #print sequence_file
            self.sequence_files.append(sequence_file)
            
    # define current file for reading
    def get_sequence(self):
        
        if self.sequence_index>=len(self.sequence_files):
            self.sequence_index = 0
            
        sequence_file = self.sequence_files[self.sequence_index]
        self.sequence_index = self.sequence_index+1
        #print sequence_file
        return self.read_sequence(sequence_file)
        
    #read sequence_file and return array of data (x, y, z, gt - label)
    def read_sequence(self, sequence_file):
        # load files with data as records
        data = numpy.load(sequence_file).view(numpy.recarray)
    
        data.gt[numpy.where(data.gt==7)] = 4
        
        # convert records with data to array with z coordinates and gt as label of class
        sequence_matrix = numpy.asarray(zip(data.z, data.gt))
  
        return sequence_matrix