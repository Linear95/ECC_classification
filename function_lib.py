#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

import tensorflow as tf
import numpy as np

#sess=tf.InteractiveSession()

stddev_list=[0,0.05,0.1,0.2,0.3,0.4,0.5]

test_batch_size=100;
test_round=1000;

def add_noise(y,sd):
    return y+np.random.normal(0,sd,np.shape(y))

def hamming_accuracy(result,label):
    y_res=(result>0.5)*1
    predict=np.apply_along_axis(decoder,1,y_res)
    return np.mean((predict==label)*1)

def one_hot_accuracy(result,label):
    predict=np.apply_along_axis(np.argmax,1,result)
    return np.mean((predict==label)*1)


def one_hot(label):
    new_label = np.zeros([len(label),10])
    for i in range(len(label)):
        new_label[i,label[i]] = 1 
    return new_label

def encoder(label):
    a = [[0,0,0,0,0,0,0],[0,1,0,0,1,0,1],[1,0,0,0,0,1,1],[1,1,0,0,1,1,0], [0,0,0,1,1,1,1],[0,1,0,1,0,1,0], [1,0,0,1,1,0,0] , [1,1,0,1,0,0,1], [0,0,1,0,1,1,0], [0,1,1,0,0,1,1], [1,0,1,0,1,0,1]]
    new_label = np.zeros([len(label),7])
    for i in range(len(label)):
        new_label[i,:] = a[label[i]] 
    return new_label

def decoder(y):
    a = [[0,0,0,0,0,0,0],[0,1,0,0,1,0,1],[1,0,0,0,0,1,1],[1,1,0,0,1,1,0], [0,0,0,1,1,1,1],[0,1,0,1,0,1,0], [1,0,0,1,1,0,0] , [1,1,0,1,0,0,1], [0,0,1,0,1,1,0], [0,1,1,0,0,1,1], [1,0,1,0,1,0,1]]
    return np.argmin(abs(a-y).sum(axis=1))
  

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
