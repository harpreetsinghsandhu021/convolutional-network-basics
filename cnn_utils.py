import math
import numpy as np
import h5py 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


def load_happy_dataset():
    # loading training dataset
    train_dataset = h5py.File('datasets/train_happy.h5','r')

    # retrieves the dataset named `train_set_x` from the `train_dataset`. the slicing `[:]` means that all data is being selected, data is converted to numpy array
    # this array typically contains the features (input_data) for the training set
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) 

    # retrieves the dataset named `train_set_y` from the `train_dataset`. which contains the labels (output data) for the training set
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    # loading test dataset
    test_dataset = h5py.File('datasets/test_happy.h5','r')

    # retrieves the dataset named `test_set_x` from the `test_dataset`, which contains features for the test set, and converts into numpy array
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    
    # retrieves the dataset named `test_set_y` from the `test_dataset`, which contains features for the test set, and converts into numpy array
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    # loading class labels
    # retrieves a dataset named 'list_classes' from the 'test_dataset', which contains the names of different classes present in the dataset
    classes = np.array(test_dataset['list_classes'][:])
    
    # reshaping labels

    # reshape `train_set_y_orig` to ensure that it has two dimensions. The new shape will have one row and as many columns are there are labels. This is often done to       make sure that label arrays have consistent dimensions, especially when feeding them into machine learning models.
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) 

    # Similarly, this reshapes test_set_y_orig to have 1 row and as many columns as there are test labels.
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes




def load_signs_dataset():
    # loading training dataset
    train_dataset = h5py.File('datasets/train_signs.h5','r')

    # retrieves the dataset named `train_set_x` from the `train_dataset`. the slicing `[:]` means that all data is being selected, data is converted to numpy array
    # this array typically contains the features (input_data) for the training set
    train_set_x_orig = np.array(train_dataset['train_set_x'][:]) 

    # retrieves the dataset named `train_set_y` from the `train_dataset`. which contains the labels (output data) for the training set
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])

    # loading test dataset
    test_dataset = h5py.File('datasets/test_signs.h5','r')

    # retrieves the dataset named `test_set_x` from the `test_dataset`, which contains features for the test set, and converts into numpy array
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])
    
    # retrieves the dataset named `test_set_y` from the `test_dataset`, which contains features for the test set, and converts into numpy array
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])

    # loading class labels
    # retrieves a dataset named 'list_classes' from the 'test_dataset', which contains the names of different classes present in the dataset
    classes = np.array(test_dataset['list_classes'][:])
    
    # reshaping labels

    # reshape `train_set_y_orig` to ensure that it has two dimensions. The new shape will have one row and as many columns are there are labels. This is often done to       make sure that label arrays have consistent dimensions, especially when feeding them into machine learning models.
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) 

    # Similarly, this reshapes test_set_y_orig to have 1 row and as many columns as there are test labels.
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes




def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y
