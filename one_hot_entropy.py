from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

import tensorflow as tf
import numpy as np
import function_lib as fl

sess=tf.InteractiveSession()

def one_hot(label):
    new_label = np.zeros([len(label),10])
    for i in range(len(label)):
        new_label[i,label[i]] = 1 
    return new_label

# def encoder(label):
    # a = [[0,0,0,0,0,0,0],[0,1,0,0,1,0,1],[1,0,0,0,0,1,1],[1,1,0,0,1,1,0], [0,0,0,1,1,1,1],[0,1,0,1,0,1,0], [1,0,0,1,1,0,0] , [1,1,0,1,0,0,1], [0,0,1,0,1,1,0], [0,1,1,0,0,1,1], [1,0,1,0,1,0,1]]
    # new_label = np.zeros([len(label),7])
    # for i in range(len(label)):
        # new_label[i,:] = a[label[i]] 
    # return new_label

# def decoder(y):
    # a = [[0,0,0,0,0,0,0],[0,1,0,0,1,0,1],[1,0,0,0,0,1,1],[1,1,0,0,1,1,0], [0,0,0,1,1,1,1],[0,1,0,1,0,1,0], [1,0,0,1,1,0,0] , [1,1,0,1,0,0,1], [0,0,1,0,1,1,0], [0,1,1,0,0,1,1], [1,0,1,0,1,0,1]]
    # return np.argmin(abs(a-y).sum(axis=1))
  

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

#___________________________one hot construction________________________________________________

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

x_image=tf.reshape(x,[-1,28,28,1])

h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

W_fc1=weight_variable([7*7* 64, 1024])
b_fc1=bias_variable([1024])

h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

y_conv=tf.matmul(h_fc1_drop,W_fc2)+b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#output=tf.nn.sigmoid(y_conv)

#MSE=tf.nn.l2_loss(y_-y_conv)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

##correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def add_noise(y,sd):
    return y+np.random.normal(0,sd,np.shape(y))

# def hamming_accuracy(result,label):
    # y_res=(result>0.5)*1
    # predict=np.apply_along_axis(decoder,1,y_res)
    # return np.mean((predict==label)*1)

def one_hot_accuracy(result,label):
    predict=np.apply_along_axis(np.argmax,1,result)
    return np.mean((predict==label)*1)

print('one_hot_entropy begins')
	
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if (i+1) % 100 == 0:
            y0_result = sess.run(y_conv,feed_dict={ x: batch[0],keep_prob:1.0})
            print('step %d, training accuracy: %g' % (i+1,one_hot_accuracy(y0_result,batch[1])))
        train_step.run(feed_dict={x: batch[0], y_: one_hot(batch[1]), keep_prob: 0.5})
    res=np.zeros(10)
    for i in range(10):
        acy=0
        for _ in range(fl.test_round):
            test_batch=mnist.test.next_batch(fl.test_batch_size)
            y0_result=sess.run(y_conv,feed_dict={x:add_noise(test_batch[0],i/20.),keep_prob:1.0})
            acy=acy+one_hot_accuracy(y0_result,test_batch[1])
        print('test accuracy with noise sd %g: one_hot %g' % (i/20.,acy/1./fl.test_round))
        res[i]=acy/1./fl.test_round
    print(res)		
		
    # for i in [0,0.05,0.1,0.2,0.3,0.4,0.5]:
        # acy=0
        # for _ in range(100):
            # test_batch=mnist.test.next_batch(100)
            # y0_result=sess.run(y_conv,feed_dict={x:add_noise(test_batch[0],i),keep_prob:1.0})
            # acy=acy+one_hot_accuracy(y0_result,test_batch[1])
        # print('test accuracy with noise sd %g: one_hot %g' % (i,acy/100.))

