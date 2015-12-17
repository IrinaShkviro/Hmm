# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 14:20:42 2015

@author: Ren
"""

__docformat__ = 'restructedtext en'

import os
import numpy
import matplotlib.pyplot as plt
        

def vis_log_reg(base_folder,
                output_folder,
                train_cost,
                train_error,
                valid_error,
                window_size,
                learning_rate):
    print "Visualizer visualize_costs"
        
    if not os.path.isdir(base_folder):
        os.makedirs(base_folder)
    os.chdir(base_folder)
                
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    print('Set output')
                        
    train_cost=numpy.asarray(train_cost).mean(axis = 0)
    train_error=numpy.asarray(train_error).mean(axis = 0)
    valid_error=numpy.asarray(valid_error).mean(axis = 0)
    print('converted to arrays')
                
    # print errors
    plt.figure(1)
    plt.plot(train_error[:, 0],train_error[:,1],label='train_error')
    plt.plot(valid_error[:, 0],valid_error[:,1],label='valid_error')
    print('plots created, start decor')        
        
    # decorative part       
    plt.xlabel('epochs')
    plt.ylabel('error(%)')
    plt.title(
        ('WS: %i  LR: %f')
        % (window_size, learning_rate)
    )
    plt.legend(loc='upper left')
    plot_name = ('error LR %f WS %i.png')%(learning_rate, window_size)
    plt.savefig(plot_name, dpi=200)
    plt.close()
    print('errors visualized')
        
    # print cost
    plt.figure(2)
    plt.plot(train_cost[:, 0],train_cost[:,1],label='train_cost')

    # decorative part      
    plt.xlabel('epochs')
    plt.ylabel('cost')
    plt.title(
        ('Window size: %i  Learning rate: %f')
        % (window_size, learning_rate)
    )
    plt.legend(loc='upper right')
    plot_name = ('cost LR %f WS %i.png')%(learning_rate, window_size)
    plt.savefig(plot_name, dpi=200)                    
    plt.clf()
    plt.close()
    print('cost visualized')
        
    os.chdir('../')
    os.chdir('../')

def visualize_pretraining(train_cost, window_size, learning_rate, corruption_level,
                          n_hidden, da_layer, datasets_folder, base_folder):
        print "Visualizer visualize_sda_costs"
                    
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        os.chdir(base_folder)
        
        if not os.path.isdir(datasets_folder):
            os.makedirs(datasets_folder)
        os.chdir(datasets_folder)
        print('Set output')
        
        example_folder = ('WS %f')%(window_size)
        if not os.path.isdir(example_folder):
            os.makedirs(example_folder)
        os.chdir(example_folder)
                        
        train_cost=numpy.asarray(train_cost)
        print('converted to array')      
        
        # print errors
        plt.figure(1)
        plt.plot(train_cost[:, 0],train_cost[:,1],label='train_cost')
        print('plot created, start decor')
        
        # decorative part       
        plt.xlabel('epochs')
        plt.ylabel('cost')
        if learning_rate<=0:
            plt.title(
                ('CL: %f Hid: %i')
                % (corruption_level, n_hidden)
            )
        else:
            plt.title(
                ('LR: %f CL: %f Hid: %i')
                % (learning_rate, corruption_level, n_hidden)
            )
        plt.legend(loc='upper left')
        if learning_rate<=0:
            plot_name = ('Pretrain layer %i CL %f Hid %i.png') \
                % (da_layer, corruption_level, n_hidden)
        else:
            plot_name = ('Pretrain layer %i LR %f CL %f Hid %i.png') \
                % (da_layer, learning_rate, corruption_level, n_hidden)
        plt.savefig(plot_name, dpi=200)
        plt.close()
        print('costs visualized')
        
        os.chdir('../')
        os.chdir('../')
        os.chdir('../')
        
def visualize_finetuning(train_cost, train_error, valid_error, test_error, 
                    window_size, learning_rate, datasets_folder, base_folder):
        print "Visualizer visualize finetuning costs"
               
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        os.chdir(base_folder)
        
        if not os.path.isdir(datasets_folder):
            os.makedirs(datasets_folder)
        os.chdir(datasets_folder)
        print('Set output')
        
        example_folder = ('WS %f')%(window_size)
        if not os.path.isdir(example_folder):
            os.makedirs(example_folder)
        os.chdir(example_folder)
                                                                
        train_cost=numpy.asarray(train_cost)
        train_error=numpy.asarray(train_error)
        valid_error=numpy.asarray(valid_error)
        test_error=numpy.asarray(test_error)
        print('converted to arrays')
                
        # print errors
        plt.figure(1)
        plt.plot(train_error[:, 0],train_error[:,1],label='train_error')
        plt.plot(valid_error[:, 0],valid_error[:,1],label='valid_error')
        plt.plot(test_error[:, 0],test_error[:,1],label='test_error')
        print('plots created, start decor')        
        
        # decorative part       
        plt.xlabel('epochs')
        plt.ylabel('error(%)')
        if learning_rate<=0:
            plt.title(
                ('WS: %i')
                % (window_size)
            )
            plot_name = ('error WS %i.png')%(window_size)
        else:
            plt.title(
                ('WS: %i  LR: %f')
                % (window_size, learning_rate)
            )
            plot_name = ('error LR %f WS %i.png')%(learning_rate, window_size)
        plt.legend(loc='upper left')
        plt.savefig(plot_name, dpi=200)
        plt.close()
        print('errors visualized')
        
        # print cost
        plt.figure(2)
        plt.plot(train_cost[:, 0],train_cost[:,1],label='train_cost')

        # decorative part      
        plt.xlabel('epochs')
        plt.ylabel('cost')
        if learning_rate<=0:
            plt.title(
                ('Window size: %i')
                % (window_size)
            )
            plot_name = ('cost WS %i.png')%(window_size)
        else:
            plt.title(
                ('Window size: %i  Learning rate: %f')
                % (window_size, learning_rate)
            )
            plot_name = ('cost LR %f WS %i.png')%(learning_rate, window_size)

        plt.legend(loc='upper right')
        plt.savefig(plot_name, dpi=200)                    
        plt.clf()
        plt.close()
        print('cost visualized')
        
        os.chdir('../')
        os.chdir('../')
        os.chdir('../')

def test_visualizer():
    print('test')
    train_cost = []
    for i in xrange(10):
        train_cost.append([])
        train_cost[-1].append(i)
        train_cost[-1].append(i*i)

    valid_cost = []
    for i in xrange(10):
        valid_cost.append([])
        valid_cost[-1].append(i)
        valid_cost[-1].append(i*i*i)
        
    test_cost = []
    for i in xrange(10):
        test_cost.append([])
        test_cost[-1].append(i)
        test_cost[-1].append(2*i)
        
    train_cost=numpy.asarray(train_cost)
    valid_cost=numpy.asarray(valid_cost)
    test_cost=numpy.asarray(test_cost)
       
    print(train_cost)
    
    f = open('train_array.txt', 'w')
    train_cost.tofile(f)
    f.close()    
    # print costs
    plt.figure(1)
    plt.plot(train_cost[:, 0],train_cost[:,1],label='x^2')
    plt.plot(valid_cost[:, 0],valid_cost[:,1],label='x^3')
    plt.plot(test_cost[:, 0],test_cost[:,1],label='2x')
        
    # decorative part       
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Some simple functions')
    plt.legend(loc='upper left')
         
    plt.savefig('test.png', dpi=200)               
    plt.clf()
    plt.close()
        
if __name__ == '__main__':
    test_visualizer()
