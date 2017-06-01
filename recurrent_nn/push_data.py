# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:42:50 2017

@author: audiffren
"""

from __future__ import print_function

import tensorflow as tf
import time
from LSTM import MultiLayerLSTMTF
from db.import_twitter_db import Twitter_db
import numpy as np
   




batch_size=50
save_error_every=10

max_length=30

epochs = 100000

laba=1e-5

dropout=0.5
eta = 0.01
load_data = False


start_time=time.time()


twitter_db = Twitter_db()

train, test = twitter_db.get_db()

def load_session(session):
    saver = tf.train.Saver()
    # Restore variables from disk.
    saver.restore(session, "deep_saves/sentiment_net.ckpt")
    print("Model restored.")
    
def save_lstm(session):
        saver = tf.train.Saver()
        saver.save(sess, 'deep_saves/sentiment_net.ckpt')  
        print("saved successful")
        
        

        

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length
    


def last_relevant(output, local_length):
    local_batch_size = tf.shape(output)[0]
    local_max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, local_batch_size) * local_max_length + (local_length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant
    
    
"""
multi-layer
"""

if __name__ == '__main__':
    
    import os
    
    if os.uname()[1]=="anguille" :
        os.system('scp "%s" "%s:%s"' % ("sentiment_analysis.py", "lu", "/home/audiffren/python/deep/sentiment_analysis_xp/sentiment_analysis.py" ) )
        os.system('scp -r "%s" "%s:%s"' % ("db/", "lu", "/home/audiffren/python/deep/sentiment_analysis_xp/db/" ) )
        print("file copied")
        
    else :
        
    
        text_x,text_y = twitter_db.get_test_set()
        with tf.variable_scope("train"):
            ndim=twitter_db.get_db_dim()
            
            # Data
            x_= tf.placeholder(tf.float64,shape=[None,max_length,ndim])
            
            y_ = tf.placeholder(tf.float64,shape=[None,1])
            
            l_ =  length(x_)
            
            keep_param = tf.placeholder(tf.float64)
            
            temp = tf.placeholder(tf.float64)
            
            
               
            #Create the network
            #Multi-layer
            layer_1 = MultiLayerLSTMTF(dim_input=ndim,dim_output=1,hidden_dims=[20],keep_param=keep_param)
            
            y1,ya1, yb1 = layer_1.feed_signal(x_,length=l_)
            
            last = last_relevant(output=y1,local_length=l_)
            
            
            capped_last= tf.clip_by_value(last, -10., 10.)
            
            logistic_output = 1./(1.+tf.exp(-capped_last/temp))
            
            
            
            
                
            # Error
            positif_instances = tf.equal(y_,1)
            negative_instances = tf.equal(y_,-1)
            
            npl = tf.zeros_like(y_)
            npl = tf.where(condition=positif_instances,x=tf.log(logistic_output),y=npl)
            npl = tf.where(condition=negative_instances,x=tf.log(1-logistic_output),y=npl)
            
            if tf.__version__ == '0.10.0':
                cross_entropy = -tf.reduce_mean( npl)
            else :
                cross_entropy = -tf.reduce_mean( npl,axis=0)
            # Training
                
            tv = tf.trainable_variables()
            regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
                
            cross_entropy+=regularization_cost*laba
            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
            
            gvs = optimizer.compute_gradients(cross_entropy)
            capped_gvs = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs]
            
            train_step = optimizer.apply_gradients(capped_gvs)
            
            #grad_check = tf.check_numerics(capped_gvs)
            #with tf.control_dependencies([grad_check]):
                
            
            
            
            
          
          
            
                
            errors = []
            
            # Algorithm ---- First training step
            
            
                
            
            if tf.__version__ == '0.10.0':
                init = tf.initialize_all_variables()   
            else :
                init = tf.global_variables_initializer()
            
            sess = tf.Session()
            sess.run(init)
            
            
            
        if load_data:
            print("Loading")
            load_session(sess)
        else :
            print("Training")
        
            for batch in range(epochs):
                batch_x,batch_y = twitter_db.get_next_batch(batch_size)
                if batch%save_error_every== 0:
                    acc_train = sess.run(cross_entropy,feed_dict={x_:batch_x,y_:batch_y,keep_param:1.,temp:.5})
                    print("Step",batch,"cross_entropy TRAIN",acc_train)
                    if batch%(10*save_error_every)== 0:
                        acc_test = sess.run(cross_entropy,feed_dict={x_:text_x,y_:text_y,keep_param:1.,temp:.5})
                        print("-----------------------------------------Step",batch,"cross_entropy TEST",acc_test)
                        errors.append([batch,acc_test,acc_train])
                        if batch%(100*save_error_every)==0:
                            save_lstm(sess)
                sess.run(train_step, feed_dict={x_:batch_x,y_:batch_y,keep_param:dropout,temp:1.})
         
            save_lstm(sess)
        
        
        #sess.close()
        print("All done, time elapsed",time.time()-start_time)
        
        
        
    