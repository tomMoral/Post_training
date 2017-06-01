# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#   Modified by Julien Audiffren
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time


import numpy as np
import tensorflow as tf


flags = tf.flags
logging = tf.logging
savepath ="saves/save.chk"
flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")

FLAGS = flags.FLAGS

def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def dropout(x,keep_param):
    return tf.nn.dropout(x,keep_prob=keep_param)
    
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def data_type():
  return  tf.float32


def extract_subimages(image,step=2,size=32):
    l=[]
    for i in range(size,image.shape[0],step):
        for j in range(size,image.shape[1],step):
            subimage=image[i-size:i,j-size:j]
            l.append(subimage)
    
    return l
    
def get_rotations(image):
    return [np.rot90(image,k=1),image,np.rot90(image,k=-1)]
    
    
def multisample(image,label,step=2,size=32,rotation=True):
    images = extract_subimages(image,step=step,size=size)
    if rotation:
        nimages=[]
        for i in images :
            nimages+=get_rotations(i)
    else : 
        nimages=images
    return nimages,[label]*len(nimages)
    

class FACESInput(object):
    def __init__(self, config, name=None):
        x,y =  get_faces_data()
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for i in range(40):
            for j in range(i*10,i*10+9):
                images,labels = multisample(x[j],y[j],rotation=False)
                x_train+=images
                y_train+=labels
            j=i*10+9
            images,labels = multisample(x[j],y[j],rotation=False)
            x_test+=images
            y_test+=labels
        train = zip(x_train,y_train)
        np.random.shuffle(train)
        self.train = train
        self.index=0
        self.test = [np.array(x_test),np.array(y_test)]
        
        self.batch_size =  config.batch_size
        self.num_steps =  config.num_steps
        self.epoch_size = len(train) // self.batch_size
        
    def set_new_config(self,config):
        self.batch_size =  config.batch_size
        self.num_steps =  config.num_steps
        self.epoch_size = len(self.train) // self.batch_size
        
        
    
    def get_next_batch(self) :
        if self.index+self.batch_size>len(self.train):
            self.index=0
        
        self.index+=self.batch_size
        data = self.train[self.index-self.batch_size:self.index]
        data = zip(*data)
        return np.array(data[0]),np.array(data[1])
    
    def get_test_data(self):
        return self.test

        

def get_faces_data():
    import pickle as pkl
    try :
        f= open("data_faces.pkl","r+")
        data = pkl.load(f)
        f.close()
        print("loaded")
        return data 
    except Exception as e:
        from sklearn.datasets import fetch_olivetti_faces
        data =  fetch_olivetti_faces()
        x=data["images"]        
        y=data["target"]
        f= open("data_faces.pkl","w+")
        pkl.dump([x,y],f)
        f.close()
        return [x,y]


class FACESModel(object):
  """The PTB model."""

  def __init__(self, config,x_,y_,k_):
    #is_training : 1 classic training, 0 no training  
    size = config.hidden_size
    size_cnn = config.hidden_size_conn
    
    self.config = config
   
    x_image = tf.reshape(x_,[-1,32,32,1])
    y_image = tf.one_hot(y_,depth=40,on_value=1,off_value=0)
    
    
    p_size =1
    p_input = x_image
    s_image = 32
    for s in size :
        w = weight_variable([5,5,p_size,s])
        b = bias_variable([s])
        hc = tf.nn.relu(conv2d(p_input,w)+b)
        hp = max_pool_2x2(hc)
        #hp = dropout(hp,keep_param=k_)
        p_size = s
        p_input = hp
        s_image//=2
    
    size_last_layer = p_size*(s_image*s_image)
    output = tf.reshape(p_input,[-1,size_last_layer])
    p_input=output
    
    for s in size_cnn :
        w = weight_variable([size_last_layer,s])
        b = bias_variable([s])
        hc = tf.nn.relu(tf.matmul(p_input,w)+b)
        hp = dropout(hc,keep_param=k_)
        size_last_layer = s
        p_input = hp
        
    
    softmax_w = weight_variable([size_last_layer,40 ])
    softmax_b = bias_variable( [40])
    
    
    
#    softmax_w = tf.get_variable("softmax_w", [size_last_layer,10 ], dtype=data_type())
#    softmax_b = tf.get_variable("softmax_b", [10], dtype=data_type()) 
    y_conv = tf.matmul(p_input, softmax_w) + softmax_b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_image,logits=y_conv))
    self._cost = cost = cross_entropy
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_image,1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    self._lr = tf.Variable(0.0, trainable=False)
    optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(cost)
    
    self._train_op = optimizer
    
    #KL 
    tvars_kl = [softmax_b,softmax_w]
    regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tvars_kl ])
    cost_kl = cost + config.lambda_reg*regularization_cost
    
    optimizer_kl = tf.train.RMSPropOptimizer(config.learning_rate).minimize(cost_kl,var_list=tvars_kl)
    
    self._train_op_kl = optimizer_kl
    
    

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})


  @property
  def cost(self):
    return self._cost

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
    
  @property
  def train_op_kl(self):
    return self._train_op_kl
    
  @property
  def accuracy(self):
    return self._accuracy


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 50
  num_steps = 20
  hidden_size = [32]
  hidden_size_conn = [512]
  max_epoch = 4
  max_max_epoch = 200
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 50
  lambda_reg = 1e-2
  name = "error_faces_small_config"


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 50
  num_steps = 20
  hidden_size = [32,64]
  hidden_size_conn = [512]
  max_epoch = 4
  max_max_epoch = 200
  keep_prob = .7
  lr_decay = .7
  batch_size = 50
  lambda_reg = 1e-2
  name="error_faces_medium_config"
  
class LargeConfig(object):
  """Large config."""
  init_scale = 0.1
  learning_rate = 1e-2
  max_grad_norm = 50
  num_steps = 20
  hidden_size = [32,64]
  hidden_size_conn = [1024]
  max_epoch = 4
  max_max_epoch = 200
  keep_prob = .5
  lr_decay = 0.9
  batch_size = 100
  lambda_reg = 10
  name = "error_faces_large_config"




def run_epoch(session, model, input,x_,y_,k_, eval_op=None, verbose=False,train=True,last_kernel=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  accuracy = 0.0
  keep_param = 1 if (not train) or last_kernel  else model.config.keep_prob
  fetches = {
      "cost": model.cost,
      "accuracy": model.accuracy
  }
  
  if train and  eval_op is not None:
    fetches["eval_op"] = eval_op

  number_of_step = 100 if train else 1
  
  for step in range(1,number_of_step+1):
        if train and eval_op is not None :
          batch_x,batch_y = input.get_next_batch()
        else :
          batch_x,batch_y = input.get_test_data()
        feed_dict = {x_:batch_x,y_:batch_y,k_:keep_param}
        
        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        accuracy += vals["accuracy"]
    
        costs += cost
        
    
        if number_of_step>10 and verbose and step % (number_of_step // 5) == 1:
          print("%.3f cross_entropy: %.3f accuracy:%.5f speed: %.0f wps" %
                (step * 1.0 /number_of_step, costs / step, accuracy/step,
                 step * model.config.batch_size / (time.time() - start_time)))

  return accuracy/number_of_step #costs / number_of_step, accuracy/number_of_step


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)




if __name__ == "__main__":
    
    import os
    
    if  0 and os.uname()[1]=="anguille" :
        print("copy files ?")
        if input():
            os.system('scp "%s" "%s:%s"' % ("faces.py", "lu", "/home/audiffren/python/deep/medium_conv_xps/faces.py" ) )
            print("files copied")
        
        print("get files ?")
        if input():
            try :
                os.system('scp "%s:%s" "%s" ' % ( "lu", "/home/audiffren/python/deep/medium_conv_xps/error_faces_small_config", "error_faces_small_config.txt" ) )
                os.system('scp "%s:%s" "%s" ' % ( "lu", "/home/audiffren/python/deep/medium_conv_xps/error_faces_medium_config", "error_faces_medium_config.txt" ) )
                os.system('scp "%s:%s" "%s" ' % ( "lu", "/home/audiffren/python/deep/medium_conv_xps/error_faces_large_config", "error_faces_large_config.txt" ) )
                print("data recovered")
            except :
                pass
            
        print("plot fig?")
        if input():
            import json
            
            for fname in ["error_faces_small_config","error_faces_medium_config","error_faces_large_config"]:
            
                try : 
                    f= open(fname+".txt","rt+")
                    error= json.load(f)
                    error= np.array(error).T
                    epochs,train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2 = error
                    f.close()
                    
                    import pylab as plb
                    
                    
                    f,a = plb.subplots(2,1,sharex=True)
                    a[0].semilogx(epochs,1-train_perplexity ,c='b',label="Regular")
                    a[0].semilogx(epochs,1-train_perplexity_2 ,c='r',label="LK")
                    a[0].legend()
                    a[0].set_title("Train perplexity")
                    a[0].set_ylabel("Classification Error")
                
                    a[1].semilogx(epochs,1-valid_perplexity,c='b')
                    a[1].semilogx(epochs,1-valid_perplexity_2 ,c='r')
                    a[1].legend()
                    a[1].set_title("Test perplexity")
                    a[1].set_ylabel("Classification Error")
                    a[1].set_xlabel("Epochs")
                    
                    
                    plb.savefig(fname + ".png")
                    plb.show()
                except Exception as e:
                    print(e)
        
        
        
    else :
      errors = []
      config = get_config()
      speak=False
      if True :
          x_ = tf.placeholder(tf.float32, shape=[None, 32,32])
          y_ = tf.placeholder(tf.int32, shape=[None])
          k_ = tf.placeholder(tf.float32)
        
    
          train_input = FACESInput(config)
          with tf.variable_scope("Model", reuse=None):
            
            m = FACESModel( config=config,x_=x_,y_=y_,k_=k_)
          
          with tf.variable_scope("Model", reuse=1):
              with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for i in range(config.max_max_epoch):
                    saver = tf.train.Saver()
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)
            
                    if speak :
                            variables_names =[v.name for v in tf.trainable_variables()]
                            values = session.run(variables_names)
                            data_0 = zip(variables_names, values)     
                    
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity = run_epoch(session, m,input=train_input,x_=x_,y_=y_,k_=k_, eval_op=m.train_op,
                                                 verbose=True,train=True,last_kernel=False)
                    print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
                    
                    
                    
                    if speak :
                            variables_names =[v.name for v in tf.trainable_variables()]
                            values = session.run(variables_names)
                            data = zip(variables_names, values)  
                            for d in range(len(data)) :
                                if not np.array_equal(data_0[d][1],data[d][1]):
                                    print("Updated coefs for "+data[d][0])
    
                    
                    
                    saver.save(session, savepath)
                    
                    
                    valid_perplexity = run_epoch(session, m,input=train_input,x_=x_,y_=y_,k_=k_, eval_op=m.train_op,
                                                 verbose=True,train=False)
                    print("Epoch: %d Valid Accuracy: %3f" % (i + 1, valid_perplexity))
                    
                    if speak :
                            variables_names =[v.name for v in tf.trainable_variables()]
                            values = session.run(variables_names)
                            data12 = zip(variables_names, values)  
                            for d in range(len(data)) :
                                if not np.array_equal(data[d][1],data12[d][1]):
                                    print("WARNIGNGINGINGINGNIIGNGNINI --------------------Updated coefs for "+data[d][0])
                    
                    train_perplexity_2 = run_epoch(session, m,input=train_input,x_=x_,y_=y_,k_=k_, eval_op=m.train_op_kl,
                                                 verbose=True,train=True,last_kernel=True)
                    print("Epoch: %d Train Perplexity: %.3f for Last Kernel" % (i + 1, train_perplexity_2))
                    
                    
                    if speak :
                            variables_names =[v.name for v in tf.trainable_variables()]
                            values = session.run(variables_names)
                            data2 = zip(variables_names, values)  
                            for d in range(len(data)) :
                                if not np.array_equal(data[d][1],data2[d][1]):
                                    print("Updated coefs for "+data[d][0])
                            
                    
                    
                    
                    
                    valid_perplexity_2 = run_epoch(session, m,input=train_input,x_=x_,y_=y_,k_=k_, eval_op=m.train_op,
                                                 verbose=True,train=False)
                    print("Epoch: %d Valid Perplexity: %.3f for Last Kernel" % (i + 1, valid_perplexity_2))
                    
                
                    saver.restore(session, savepath )
                    
                    variables_names =[v.name for v in tf.trainable_variables()]
                    values = session.run(variables_names)
                    data3 = zip(variables_names, values)  
                    if speak :
                            for d in range(len(data)) :
                                if not np.array_equal(data2[d][1],data3[d][1]):
                                    if np.array_equal(data[d][1],data3[d][1]):
                                        print("Reverted coefs for "+data[d][0])         
                                    else :
                                        print("Updated coefs for "+data[d][0])         
                        
                    errors.append([i+1,train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2])
                    print("------------------------------------------------------------------------")
                    
                
                
                

      import json
      f= open(config.name,"wt+")
      json.dump(errors,f)
      f.close()
            
#            if FLAGS.save_path:
#                    print("Saving model to %s." % FLAGS.save_path)
#                    sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)
#                        

        

    

