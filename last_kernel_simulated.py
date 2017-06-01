# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:47:03 2016

@author: audiffren
"""
from __future__ import print_function

import tensorflow as tf
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
font = {'family' : 'serif',
       'weight' : 'bold',
      'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
import pylab as plb
import numpy as np
import time
from  tensorflow.python import linalg_ops as ops

import argparse
# Parse program arguments
try :
    parser = argparse.ArgumentParser(description='Xp deep regression last kernel')
    
    parser.add_argument('xp_dim', metavar='xp_dim', type=int, nargs='+',
                       help='Number of the XP')
    args = parser.parse_args()
    xp_dim =args.xp_dim[0]
except :
    xp_dim=0
print('cp dim',xp_dim)

batch_size=50
simulated=True

load_figure =  False
epochs = 700
epochs_last_kernel = 250

save_error_every=10
save_true_error_every=10

trajectory_each = 50
trajectory_starting = 0

laba=10.0


start_time=time.time()

batch_index=0
train=None

x_test=None
y_test=None


dic_var_save = {'batch_size':batch_size,
'simulated':simulated,
'epochs':epochs,
'cp dim':xp_dim,
'save_error_every':save_error_every,
'save_true_error_every':save_true_error_every,
'laba':laba}

def load_data():
    import pickle as pkl
    f = open("DNN saves/data_saved_simulated.json","rt+")
    a = pkl.load(f)
    f.close()
    return a
    


if load_figure :
    errors,true_errors,error_basic,true_error_basic,value_list = load_data()
else :
        
    def load_dataset():
        global train
        global x_test
        global y_test
        global indexes
        print("loading train set")
        f=np.loadtxt('parkinsons_updrs.data.txt',delimiter=",",skiprows=2)[:,3:]
        x_train = f[:,2:]
        y_train = f[:,xp_dim:1+xp_dim]
        
        print('normalizing')
        mx=np.mean(x_train,axis=0)
        x_train-=mx
        nx= np.sqrt(np.sum(x_train**2)/len(y_train))
        x_train/=nx
        
        my=np.mean(y_train,axis=0)
        y_train-=my
        ny= np.sqrt(np.sum(y_train**2)/len(y_train))
        y_train/=ny
        
        train=np.hstack([x_train,y_train])
        
        test=train[int(len(train)*0.8):]
        train=train[:int(len(train)*0.8)]
        
        
        x_test = test[:,:-1]
        y_test = test[:,-1:]
        
        
    def load_simulated_dataset(normalize = False):
        global train
        global x_test
        global y_test
        global indexes
        print("generating train set")
        weights = np.random.uniform(-1,1,[10,5])
        weights_2 = np.random.uniform(-1,1,[5,1])
        x_train = np.random.uniform(-1,1,[10000,10])
        y_train = (np.dot( np.tanh(np.dot(x_train,weights)),weights_2))
        
        if normalize :
            print('normalizing')
            mx=np.mean(x_train,axis=0)
            x_train-=mx
            nx= np.sqrt(np.sum(x_train**2)/len(y_train))
            x_train/=nx
            
            my=np.mean(y_train,axis=0)
            y_train-=my
            ny= np.sqrt(np.sum(y_train**2)/len(y_train))
            y_train/=ny
        
        train=np.hstack([x_train,y_train])
        
        test=train[int(len(train)*0.8):]
        train=train[:int(len(train)*0.8)]
        
        
        x_test = test[:,:-1]
        y_test = test[:,-1:]
        
        
    
        
        
        
    def get_next_batch():
        global batch_size
        global batch_index
        global train
        batch_index+=1
        if (batch_index)*batch_size > len(train):
            batch_index=1
            np.random.shuffle(train)
        
        
        return train[batch_size*(batch_index-1):batch_size*batch_index][:,:-1],train[batch_size*(batch_index-1):batch_size*batch_index][:,-1:]
    
    def get_all_data():
        global train
        return train[:,:-1],train[:,-1:]
    
        
    if simulated :
        load_simulated_dataset()
    else :
        load_dataset()
    ndim=train.shape[1]-1
    
    ## SHORTCUT functions
    
    
    def create_weight_variable(shape):
        initial = tf.truncated_normal(shape,mean=0.,stddev=0.1)
        return tf.Variable(initial_value=initial)
        
    def create_bias_variable(shape,trainable=True):
        initial = tf.constant(0.1,shape=shape)
        return tf.Variable(initial_value=initial,trainable=trainable)
        
        
    
    # Data
    x_= tf.placeholder(tf.float32,shape=[None,ndim])
    
    y_ = tf.placeholder(tf.float32,shape=[None,1])
    
    keep_proba = tf.placeholder(tf.float32)
    
    #Create the network
    
    W_1 = create_weight_variable([ndim,ndim])
    b_1 = create_bias_variable([ndim])
    
    h_1 = tf.nn.tanh(tf.matmul(x_,W_1) + b_1)
    h_1_drop = tf.nn.dropout(h_1,keep_prob=keep_proba)
    
    
    W_2 = create_weight_variable([ndim,10])
    b_2 = create_bias_variable([10])
    
    h_2 = tf.nn.relu(tf.matmul(h_1_drop,W_2) + b_2)
    h_2_drop = tf.nn.dropout(h_2,keep_prob=keep_proba)
    
    
    last_layer= h_2_drop # 55000 * 1024
    last_matrix = tf.matmul(last_layer,last_layer,transpose_b=True) # 55000 * 55000
    #alpha =  tf.matmul(tf.matrix_inverse( len(train)*laba*last_matrix  + tf.matmul(last_matrix,last_matrix)),tf.matmul(last_matrix,y_)) #55000 * 10
    alpha = ops.matrix_solve(last_matrix + len(train)*laba*np.identity(len(train)) ,y_ )
    final_weights = tf.matmul( last_layer,alpha,transpose_a=True ) # 1024 * 10
    
    
    
    
    W_3 = create_weight_variable([10,1])
    
    y = tf.matmul(h_2_drop,W_3)
    
    
    var_list=[W_1,W_2,W_3,b_1,b_2]
    
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(W_1)) ) + tf.sqrt(tf.reduce_sum(tf.square(W_2)) ) + tf.sqrt(tf.reduce_sum(tf.square(W_3)) ) + tf.sqrt(tf.reduce_sum(tf.square(b_1)) ) + tf.sqrt(tf.reduce_sum(tf.square(b_2)) )
    
    
    
    # Error
    
    krr =  tf.reduce_mean( tf.square(y_-y) )  + laba*norm2
    error_krr = tf.reduce_mean( tf.square(y_-y) ) 
    
    # Training
    train_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss=krr)
    train_final_step = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss=krr,var_list=[W_3])
    
    
    #load_dataset()
    
    
    errors = []
    true_errors =[]
    
    # Algorithm ---- First training step
    print("Pre train")
    error_basic = []
    true_error_basic = []
    
    init = tf.initialize_all_variables()
    
    sess = tf.Session()
    sess.run(init)
    
    total_x,total_y = get_all_data()
    
    value_list = []
    
    for batch in range(epochs):
        batch_x,batch_y = get_next_batch()
        if batch%save_error_every== 0 :
            acc = sess.run(krr,feed_dict={x_:total_x,y_:total_y,keep_proba:1.})
            #print("Step",batch,"krr",acc)
            error_basic.append([batch,acc])
        if batch%save_true_error_every == 0:
            acc = sess.run(error_krr,feed_dict={x_:x_test,y_:y_test,keep_proba:1.})
            #print("True krr",acc)
            true_error_basic.append([batch,acc])
        sess.run(train_step, feed_dict={x_:batch_x,y_:batch_y,keep_proba:1.})
        
        if batch % trajectory_each == 0 and batch>= trajectory_starting:
            value_list.append((batch,[v.eval(sess) for v in var_list]))
    
            
    
        
    
    for (batch_init,value_l) in value_list : 
        
        print("batch init time:",batch_init)
           
        # Algorithm ---- alternate end of training step
        print("Last kernel")
        error_2 = []
        true_error_2 = []
        
        
        for i,v in enumerate(var_list) :
            #load previous values
             give = v.assign(value_l[i])
             sess.run(give)
            
            
        
        for batch in range(epochs_last_kernel):
            batch_x,batch_y = get_next_batch()
            if batch%save_error_every== 0 :
                acc = sess.run(krr,feed_dict={x_:total_x,y_:total_y,keep_proba:1.})
                #print("Step",batch,"krr",acc)
                error_2.append([batch+batch_init,acc])
            if batch%save_true_error_every == 0:
                acc = sess.run(error_krr,feed_dict={x_:x_test,y_:y_test,keep_proba:1.})
                #print("True krr",acc)
                true_error_2.append([batch+batch_init,acc])
            sess.run(train_final_step, feed_dict={x_:batch_x,y_:batch_y,keep_proba:1.})
            
        
        #acc = sess.run(error_krr,feed_dict={x_:x_test,y_:y_test,keep_proba:1.})
        #acc2 = sess.run(krr,feed_dict={x_:train[:,:-1],y_:train[:,-1:],keep_proba:1.})
        #print("Test krr",acc,acc2)
        
        
        
        # Algorithm ---- Exact Solution
        print("Exact solution")
        
        for i,v in enumerate(var_list) :
             give = v.assign(value_l[i])
             sess.run(give)
            
            
        fd = final_weights.eval(session=sess,feed_dict={x_:total_x,y_:total_y,keep_proba:1.})
        
        give= W_3.assign( fd )
        sess.run( give )
        #give = b_fc_2.assign( np.zeros(10) )
        #sess.run( give )
        acc_train = sess.run(krr,feed_dict={x_:total_x,y_:total_y,keep_proba:1.})
        #print("Train krr",acc_train)
        acc_test = sess.run(error_krr,feed_dict={x_:x_test,y_:y_test,keep_proba:1.})
        #print("Test krr",acc_test)
        
        errors.append((error_2,acc_train))
        true_errors.append((true_error_2,acc_test))
    
    sess.close()
    print("All done, time elapsed",time.time()-start_time)


name_string = "img/error_train_deep_"
for name,value in dic_var_save.iteritems():
    name_string+='_'+name+'_'+str(value)
fig, ax =plb.subplots()
error_to_plot = zip(*error_basic)
ax.plot(error_to_plot[0],error_to_plot[1],c="b",label="Gradient Descent")
(error_2,acc_train) = errors[0]
error_to_plot=zip(*error_2)
ax.plot(error_to_plot[0],error_to_plot[1],c='m',label="LastKernel")
ax.plot(error_to_plot[0],[acc_train for e in error_to_plot[0]],'r--',label="optimal solution")
for (error_2,acc_train) in errors[2::2]:
    error_to_plot=zip(*error_2)
    ax.plot(error_to_plot[0],error_to_plot[1],c='m')
    ax.plot(error_to_plot[0],[acc_train for e in error_to_plot[0]],'r--')
ax.legend()
name_string += ".png"
ax.set_ylabel('Training Cost')
ax.set_xlabel('Epoch')
ax.set_title('Training Cost Simulated dataset')
fig.savefig(name_string)


name_string = "img/error_test_deep_"
for name,value in dic_var_save.iteritems():
    name_string+='_'+name+'_'+str(value)
fig, ax =plb.subplots()
error_to_plot = zip(*true_error_basic)
ax.plot(error_to_plot[0],error_to_plot[1],c="b",label="Gradient Descent")
(true_error_2,acc_test) = true_errors[0]
error_to_plot=zip(*true_error_2)
ax.plot(error_to_plot[0],error_to_plot[1],c='m',label="LastKernel")
ax.plot(error_to_plot[0],[acc_test for e in error_to_plot[0]],'r--',label="optimal solution")
for (true_error_2,acc_test) in true_errors[2::2]:
    error_to_plot=zip(*true_error_2)
    ax.plot(error_to_plot[0],error_to_plot[1],c='m')
    ax.plot(error_to_plot[0],[acc_test for e in error_to_plot[0]],'r--')
ax.legend()
name_string += ".png"
ax.set_ylabel('Test Error')
ax.set_xlabel('Epoch')
ax.set_title('Test Error Simulated dataset')
fig.savefig(name_string)




def save_data():
    import pickle as pkl
    f= open("DNN saves/data_saved_simulated.json","wt+")
    pkl.dump([errors,true_errors,error_basic,true_error_basic,value_list],f)
    f.close()
    


