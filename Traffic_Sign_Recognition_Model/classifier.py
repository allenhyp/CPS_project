
# coding: utf-8

# In[1]:


####### Module to retrieve pickled data######
from sklearn.utils import shuffle
from preprocess import load_data
########## Plotting Data ########
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import display, HTML
from tqdm import trange
import time
from datetime import datetime
import cv2

####### Mathematical & Array Operations #########
import numpy as np
import math
from random import randint
from collections import namedtuple
import itertools
import pandas as pd

# Tensor FLow for Neural Network Frameworks
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import apply_regularization
from tensorflow.contrib.layers import l1_regularizer
from tensorflow.contrib.layers import xavier_initializer
from spatial_transformer import transformer
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

tf.reset_default_graph()
########## Graph Inputs #########################
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, (None))
num_classes = 43
reg_fac = tf.placeholder(tf.float32, None)
rate = tf.placeholder(tf.float32, None)
is_training = tf.placeholder(tf.bool, None)


# In[3]:


############################# Batch Normalization Layer ##################
def batch_norm(input_, name, n_out, phase_train):
    with tf.variable_scope(name + 'bn'):
        beta = tf.Variable(tf.constant(
            0.0, shape=[n_out]), name=name + 'beta', trainable=True)
        gamma = tf.Variable(tf.constant(
            1.0, shape=[n_out]), name=name + 'gamma', trainable=True)
        if len(input_.get_shape().as_list()) > 3:
            batch_mean, batch_var = tf.nn.moments(
                input_, [0, 1, 2], name=name + 'moments')
        else:
            batch_mean, batch_var = tf.nn.moments(
                input_, [0, 1], name=name + 'moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (
            ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(
            input_, mean, var, beta, gamma, 1e-3)

    variable_summaries(beta)
    variable_summaries(gamma)
    return normed

############################## Parametric ReLU Activation  Layer #########


def parametric_relu(input_, name):
    alpha = tf.get_variable(name=name + '_alpha', shape=input_.get_shape(
    )[-1], initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.3), dtype=tf.float32)
    pos = tf.nn.relu(input_)
    tf.summary.histogram(name, pos)
    neg = alpha * (input_ - abs(input_)) * 0.5
    return pos + neg


# Convolutional Layer with activation and batc
def conv(input_, name, k1, k2, n_o, reg_fac, is_tr, s1=1, s2=1, is_act=True, is_bn=True, padding='SAME'):

    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable(name + "weights", [k1, k2, n_i, n_o], tf.float32, xavier_initializer(
        ), regularizer=tf.contrib.layers.l2_regularizer(reg_fac))
        biases = tf.get_variable(name +
                                 "bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, weights, (1, s1, s2, 1), padding=padding)
        bn = batch_norm(conv, name, n_o, is_tr) if is_bn else conv
        activation = parametric_relu(tf.nn.bias_add(
            bn, biases), name + "activation") if is_act else tf.nn.bias_add(bn, biases)
        variable_summaries(weights)
        variable_summaries(biases)
    return activation

# Fully connected Layer with activation and ba


def fc(input_, name, n_o, reg_fac, is_tr, p_fc, is_act=True, is_bn=True):
    n_i = input_.get_shape()[-1].value
    with tf.variable_scope(name):
        weights = tf.get_variable(name + "weights", [n_i, n_o], tf.float32, xavier_initializer(
        ),  regularizer=tf.contrib.layers.l2_regularizer(reg_fac))
        biases = tf.get_variable(
            name + "bias", [n_o], tf.float32, tf.constant_initializer(0.0))
        bn = tf.nn.bias_add(tf.matmul(input_, weights), biases)
        activation = batch_norm(bn, name, n_o, is_tr) if is_bn else bn
        logits = parametric_relu(
            activation, name + "activation") if is_act else activation
        
        variable_summaries(weights)
        variable_summaries(biases)

    return tf.cond(is_tr, lambda: tf.nn.dropout(logits, keep_prob=p_fc), lambda: logits)

############################# Max Pooling Layer with activation ##########


def pool(input_, name, k1, k2, s1=2, s2=2):
    return tf.nn.max_pool(input_, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding='VALID', name=name)


# In[4]:


############################# Localization Layer for Spatial Transformer L
def localization_net(input_, name, is_tr, reg_fac):
    # Identity transformation
    initial = np.array([[1., 0, 0], [0, 1., 0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    # Weight and Bias containing the identity transformation
    W = tf.get_variable('loc_weights', [64, 6], tf.float32, xavier_initializer(
    ), regularizer=tf.contrib.layers.l2_regularizer(reg_fac))
    b = tf.Variable(initial_value=initial, name='loc_bias')

    ############ Localization Network for the Spatial transformer network ####
    ##################### 7x7x16 Conv -> 2x2 Max pooling ###################
    locnet = conv(input_, name="locnet_conv1", k1=3, k2=3, n_o=16,
                  reg_fac=reg_fac, is_tr=is_tr, padding='SAME')
    locnet = pool(locnet, name="locnet_pool1", k1=2, k2=2)

    ###################### 5x5x32 Conv -> 2x2 Max pooling ###################
    locnet = conv(locnet, name="locnet_conv2", k1=3, k2=3, n_o=32,
                  reg_fac=reg_fac, is_tr=is_tr, padding='SAME')
    locnet = pool(locnet, name="locnet_pool2", k1=2, k2=2)

    ###################### 3x3x64 Conv -> 2x2 Max pooling ####################
    locnet = conv(locnet, name="locnet_conv3", k1=3, k2=3, n_o=64,
                  reg_fac=reg_fac, is_tr=is_tr, padding='SAME')
    locnet = pool(locnet, name="locnet_pool3", k1=2, k2=2)
    
    ###################### 3x3x64 Conv -> 2x2 Max pooling ####################
    locnet = conv(locnet, name="locnet_conv4", k1=3, k2=3, n_o=128,
                  reg_fac=reg_fac, is_tr=is_tr, padding='SAME')
    locnet = pool(locnet, name="locnet_pool4", k1=2, k2=2)

    ####################### Fully Connected Layers ###########################
    locnet_fc0 = flatten(locnet)
    locnet_fc1 = fc(locnet_fc0, name="locnet_fc1", n_o=128,
                    reg_fac=reg_fac, is_tr=is_tr, p_fc=.50)
    locnet_fc2 = fc(locnet_fc1, name="locnet_fc2", n_o=64,
                    reg_fac=reg_fac, is_tr=is_tr, p_fc=.50)
    locnet_op = tf.nn.bias_add(tf.matmul(locnet_fc2, W), b)

    return locnet_op


# In[5]:


def VGG_Layer(input_, name, conv_size, n_layers, pool_size, n_o, reg_fac, is_tr, p_vgg):

    n_i = input_.get_shape()[-1].value
    c_k1 = conv_size
    c_k2 = conv_size
    p_k1 = pool_size
    p_k2 = pool_size

    vgg = input_
    for i in range(n_layers):
        ############## VGG Building Block ###############################
        ############## 2 - Conv , 1- Pool 1- Dropout 1 - Batch-Norm ###########
        vgg = conv(vgg, name=name + "conv1_" + str(i), k1=c_k1,
                   k2=c_k2, n_o=n_o, reg_fac=reg_fac, is_tr=is_tr)

    vgg = pool(vgg, name=name + "pool1", k1=p_k1, k2=p_k2)
    vgg = tf.cond(is_tr, lambda: tf.nn.dropout(
        vgg, keep_prob=p_vgg), lambda: vgg)
    return vgg


# In[6]:


def run_model(input_, num_classes, params, reg_fac, is_training):

    ############## Spatial Transformer Module ######################
    ##################### Localization layer #######################
    locnet = localization_net(input_, name="locnet",
                              reg_fac=reg_fac, is_tr=is_training)
    ####################### Affine Transformation Layer ############
    stn = transformer(input_, locnet, out_size=(32, 32, 3))
    stn_= tf.reshape(stn,(-1,32,32,3))
    ################################## VGG Net ###############################
    ################################# VGG- Layer - 1 #########################
    vgg1 = VGG_Layer(stn_, "vgg1", conv_size=3, n_layers=2, pool_size=2, n_o=32,
                     reg_fac=reg_fac, is_tr=is_training, p_vgg=params.vgg1)
    variable_summaries(vgg1)
    ################################# VGG- Layer - 2 #########################
    vgg2 = VGG_Layer(vgg1, "vgg2", conv_size=3,  n_layers=2, pool_size=2, n_o=64,
                     reg_fac=reg_fac, is_tr=is_training, p_vgg=params.vgg2)
    variable_summaries(vgg2)
    ################################# VGG- Layer - 3 #########################
    vgg3 = VGG_Layer(vgg2, "vgg3", conv_size=3,  n_layers=3, pool_size=2, n_o=128,
                     reg_fac=reg_fac, is_tr=is_training, p_vgg=params.vgg3)
    variable_summaries(vgg3)
    ################################# VGG- Layer - 4 #########################
    vgg4 = VGG_Layer(vgg3, "vgg4", conv_size=3,  n_layers=3, pool_size=2, n_o=256,
                     reg_fac=reg_fac, is_tr=is_training, p_vgg=params.vgg4)
    variable_summaries(vgg4)
    ################################ Multi Scale Convolution layer ###########
    ms1 = pool(vgg2, "ms1", k1=2, k2=2)
    ms2 = tf.concat([ms1,vgg3], axis=3)
    
    ms3 = pool(ms2, "ms3", k1=2, k2=2)
    ms = tf.concat([ms3, vgg4], axis=3)

    
    ################## Fully Connected Layers for a Linear Classifier ########
    #############################First Fully Connected Layer #################
    fc0 = fc(flatten(ms), "fc0", 1024, reg_fac=reg_fac,
             is_tr=is_training, p_fc=params.fc0)
    fc1 = fc(fc0, "fc1", 512, reg_fac=reg_fac,
             is_tr=is_training, p_fc=params.fc1)
    ########################### Output readout Layer #########################
    fc2 = fc(fc1, "fc2", num_classes, reg_fac=reg_fac,
             is_tr=is_training, p_fc=1.0, is_act=False, is_bn=True)

    return fc2


# In[7]:


params = namedtuple('params', 'vgg1 vgg2 vgg3 vgg4 ms fc0 fc1')
mdltype = params(vgg1=.5, vgg2=.5, vgg3=.5, vgg4=.5, ms=.5, fc0=.5, fc1=.5)
logits = run_model(x, num_classes, mdltype, reg_fac, is_training)
tf.summary.histogram('Logits', logits)

# For Top 5 Guesses
prediction = tf.nn.softmax(logits)
top5_guesses = tf.nn.top_k(prediction, k=5, sorted=True)


# Predicted Label and Actual Label using Argmax
y_pred = tf.argmax(logits, 1)

# Accuracy Calculation
correct_prediction = tf.equal(y_pred, y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('Accuracy', accuracy_operation)

######### Cross Entropy and Loss for Training ##########
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y, name='cross_entropy')
loss_operation = tf.reduce_mean(
    cross_entropy) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
tf.summary.scalar('Loss', loss_operation)

########### Training Step ################
Train_Step = tf.train.AdamOptimizer(rate).minimize(loss_operation)
summary = tf.summary.merge_all()


# In[8]:


def evaluate(X_data, y_data, batch_size, is_tr):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        bx, by = X_data[offset:offset +
                        batch_size], y_data[offset:offset + batch_size]
        inputs_ = {x: bx, y: by, is_training: is_tr}
        accuracy = sess.run(accuracy_operation, feed_dict=inputs_)
        total_accuracy += (accuracy * len(bx))
    return total_accuracy / num_examples

def plot_confusion_matrix(X_data,y_data, batch_size, is_tr, normalize=True):
    num_examples = len(X_data)
    sess = tf.get_default_session()
    offset=randint(0,num_examples-batch_size)
    bx, by = X_data[offset:offset +
                    batch_size], y_data[offset:offset + batch_size]
    inputs_ = {x: bx, y: by, is_training: is_tr}
    cm_input = sess.run(y_pred,feed_dict=inputs_)
    cm = confusion_matrix(y_true = by,y_pred = cm_input)
    if normalize is True:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
    
    plt.figure(figsize=(25,25))  
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(num_classes)
    
    df=pd.read_csv("signnames.csv")
    plt.xticks(range(num_classes), df['SignName'])
    plt.yticks(range(num_classes), df['SignName'])
    plt.xticks(rotation=90)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel('Predicted', fontsize = 24)
    plt.ylabel('True', fontsize= 24)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


# In[10]:


# training_file = 'train_processed'
# validation_file = 'valid_processed'
# testing_file = 'test_processed'
training_file = 'train.pickle'
validation_file = 'valid.pickle'
testing_file = 'test.pickle'
X_train,y_train = load_data(training_file)
X_valid,y_valid = load_data(validation_file)
X_test,y_test = load_data(testing_file)


# In[5]:


BATCH_SIZE = 500
EPOCHS = 50
REG_FACTOR = 2e-5
RATE = 1e-4
now_datetime = datetime.now().strftime("%y%m%d")
save_file = "VGGNet_" + now_datetime
restore_file ="VGGNet_" + now_datetime
chkpt = './' + save_file
chkpt_restore = './' + restore_file
logdir = chkpt + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'
is_restore= False


# In[ ]:


init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
with sess.as_default():
    sess.run(init)
    ##################  Start Model Training  #######################
    if is_restore:
        saver.restore(sess,chkpt_restore)
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    print("Training...")
    print()
    val_acc = []
    for i in range(EPOCHS):
        
        ############ Training Operation ################
        Training_loss = 0
        X_train, y_train = shuffle(X_train, y_train)
        for offset in trange(0, len(X_train), BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            inputs = {x: batch_x, y: batch_y, reg_fac: REG_FACTOR,
                      rate: RATE, is_training: True}
            loss, _ = sess.run([loss_operation, Train_Step], feed_dict=inputs)
            Training_loss += (loss * len(batch_x))

        ################ Evaluation operation ####################
        Validation_Accuracy = evaluate(X_valid, y_valid, 1000, is_tr=False)
        val_acc.append(Validation_Accuracy)
        Training_loss /= len(X_train)
        print("Epochs:", i + 1)
        print("Training_Loss:", Training_loss)
        print("Validation_Accuracy:", Validation_Accuracy)
        
        
        ##### Save Model if the Validation accuracy gets better
        if (max(val_acc) == Validation_Accuracy):
            saver.save(sess, chkpt)
            print("Intermediate Model Save")
            summary_str = sess.run(summary, feed_dict=inputs)
            summary_writer.add_summary(summary_str, i)
            
print("Model saved")


# In[ ]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,chkpt)
    test_accuracy = evaluate(X_test, y_test, 500, False)
    plot_confusion_matrix(X_test, y_test, 2000, False)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# In[ ]:


saver=tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,chkpt)
    plot_confusion_matrix(X_test,y_test, 2000, is_tr= False, normalize=False)


# In[ ]:


import matplotlib.gridspec as gridspec
from skimage import exposure
from skimage import img_as_ubyte
from skimage import img_as_float
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import argparse
import importlib
import PreProcessing
importlib.reload(PreProcessing)
from PreProcessing import visualize_predictions, image_normalizer, images_show
from tensorflow.python.framework import graph_util


# In[ ]:


###### Load Model and Open 
def run_model(X_,image_names):
    saver=tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        total_pred,top5_pred = sess.run([prediction,top5_guesses], feed_dict={x:X_,is_training: False})
    return total_pred, top5_pred

def test_classifier(img_path):

    Images = [image_normalizer(cv2.imread(img_path+name,1)) for name in os.listdir(img_path) if (name.endswith('.png')  or name.endswith('.jpg'))]
    FileNames = [name for name in os.listdir(img_path) if ( name.endswith('.png') or name.endswith('.jpg') )]
    Images_ = img_as_float(Images)
    print("Processed Images")
    images_show(Images_,1,len(FileNames),rand=False)
    total_pred, top5_pred = run_model(Images_,FileNames)
    visualize_predictions(Images_,FileNames,total_pred, top5_pred)

