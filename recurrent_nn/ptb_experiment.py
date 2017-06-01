# Code for reproducing the PTB experiment of the paper
#
#Post training for deep learning, T. Moreau and J. Audiffren
#
#
#This code was made by J. Audiffren, and is based on the PTB code for tensorflow
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
#
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
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import time

import numpy as np
import tensorflow as tf

import reader

logging = tf.logging



def data_type():
  return  tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    #is_training : 1 classic training, 0 LK, -1 no training  
    self._input = input_
    
    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def lstm_cell():
      # With the latest TensorFlow source code (as of Mar 27, 2017),
      # the BasicLSTMCell will need a reuse parameter which is unfortunately not
      # defined in TensorFlow 1.0. To maintain backwards compatibility, we add
      # an argument check here:
      if 'reuse' in inspect.getargspec(
          tf.contrib.rnn.BasicLSTMCell.__init__).args:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True,
            reuse=tf.get_variable_scope().reuse)
      else:
        return tf.contrib.rnn.BasicLSTMCell(
            size, forget_bias=0.0, state_is_tuple=True)
    attn_cell = lstm_cell
    if is_training>0 and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
            lstm_cell(), output_keep_prob=config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell(
        [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, input_.input_data)

    if is_training>0 and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of models/tutorials/rnn/rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    # inputs = tf.unstack(inputs, num=num_steps, axis=1)
    # outputs, state = tf.contrib.rnn.static_rnn(
    #     cell, inputs, initial_state=self._initial_state)
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)

    output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    
    apply_mods = zip(grads, tvars)
    
    self._train_op = optimizer.apply_gradients(
        apply_mods,
        global_step=tf.contrib.framework.get_or_create_global_step())
    
    #KL; post training 
    tvars_kl = [softmax_b,softmax_w]
    regularization_cost = tf.reduce_sum([ tf.nn.l2_loss(v) for v in tvars_kl ])
    cost_kl = cost + config.lambda_reg*regularization_cost
    grads_kl = tf.gradients(cost_kl, tvars_kl)
    apply_mods_kl = zip(grads_kl, tvars_kl)
    
    self._train_op_kl = optimizer.apply_gradients(
        apply_mods_kl,
        global_step=tf.contrib.framework.get_or_create_global_step())        
    
    

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
    
  @property
  def train_op_kl(self):
    return self._train_op_kl




class LearningConfig(object):
  """config for the network"""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_iteration = 450
  keep_prob = 0.35
  lr_decay = (1 / 1.15)
  batch_size = 20
  vocab_size = 10000
  lambda_reg = 1e-2




def run_epoch(session, model, eval_op=None, verbose=False,last_kernel=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  number_of_batches= 100
      
  for step in range(number_of_batches):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if 0 and verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def get_config():
  return LearningConfig()


def main(_):
   
      errors = []
      data_path = "simple-examples/data/"
      save_dir = "saves/"
      raw_data = reader.ptb_raw_data(data_path)
      train_data, valid_data, test_data, _ = raw_data
    
      config = get_config()
      eval_config = get_config()
      eval_config.batch_size = 1
      eval_config.num_steps = 1
    
      with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
    
        with tf.name_scope("Train"):
          train_input = PTBInput(config=config, data=train_data, name="TrainInput")
          with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = PTBModel(is_training=1, config=config, input_=train_input)
    
        with tf.name_scope("Valid"):
          valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mvalid = PTBModel(is_training=0, config=config, input_=valid_input)
          
          
          
    
        with tf.name_scope("Test"):
          test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
          with tf.variable_scope("Model", reuse=True, initializer=initializer):
            mtest = PTBModel(is_training=0, config=eval_config,
                             input_=test_input)
    
        sv = tf.train.Supervisor(logdir=save_dir)
        with sv.managed_session() as session:
            for i in range(int(config.max_iteration)):
                epoch = (i * 100)// train_input.epoch_size
                lr_decay = config.lr_decay ** max(epoch + 1 - config.max_epoch, 0.0)
                m.assign_lr(session, config.learning_rate * lr_decay)
        
                     
                
                print("Epoch: %d Iteration: %d Learning rate: %.3f" % (epoch+1, i + 1, session.run(m.lr)))
                train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                             verbose=True)
                print("Train Perplexity: %.3f" % (train_perplexity))
                
                
                save_path = sv.saver.save(session, 'saves/model.ckpt', global_step=sv.global_step)
                
                
                valid_perplexity = run_epoch(session, mvalid)
                print("Valid Perplexity: %.3f" % (valid_perplexity))
                
                train_perplexity_2 = run_epoch(session, m, eval_op=m.train_op_kl,
                                             verbose=True,last_kernel=True)
                print("Train Perplexity: %.3f for Last Kernel" % (train_perplexity_2))
                
                
                
                valid_perplexity_2 = run_epoch(session, mvalid)
                print("Valid Perplexity: %.3f for Last Kernel" % (valid_perplexity_2))
                
            
                sv.saver.restore(session, save_path )
       
                
                errors.append([i+1,train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2])
                print("------------------------------------------------------------------------")
                
                
                
                

            import json
            f= open("error.txt","wt+")
            json.dump(errors,f)
            f.close()
            
            test_perplexity = run_epoch(session, mtest)
            print("Test Perplexity: %.3f" % test_perplexity)
            
    
            print("Saving model to %s." % save_dir)
            sv.saver.save(session, save_dir, global_step=sv.global_step)
                        


if __name__ == "__main__":
    
    
    print("run experiment ?")
    if input():
        tf.app.run()
        print("experiment finished")
    
    print("plot fig?")
    if input():
   
        
        import json
        f= open("error.txt","rt+")
        error= json.load(f)
        error= np.array(error).T
        epochs,train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2 = error
        f.close()
        
        import pylab as plb
        
        def smooth_curves(curves,param,max_val):
            new_curves=[]
            for curve in curves :
                curve=curve[:max_val]
                c=0
                for p in range(param):
                    c+=curve[p:-(param-p)]
                c/=param
                new_curves.append(c)
            return new_curves
            
        param=10
        max_val=400
        train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2 = smooth_curves([train_perplexity,valid_perplexity,train_perplexity_2,valid_perplexity_2],param,max_val)
        epochs=epochs[:max_val]
        epochs=epochs[param//2:-(param-param//2)]
            
        m1 = min(train_perplexity_2)-1
        m2= min(valid_perplexity_2)-1
        
        f,a = plb.subplots(2, 1, sharex=True, figsize=[10.0, 6.0])
        a[0].loglog(epochs[1:], train_perplexity[1:] - m1, "b--",
                    label="Regular Training")
        a[0].loglog(epochs[1:], train_perplexity_2[:-1] - m1, 'r',
                    label="Post Training")
        a[0].legend(fontsize="x-large")
        a[0].set_ylabel("Train Perplexity (- {:0.1f})".format(m1),fontsize="x-large")
        a[0].set_xlim(epochs[1],epochs[-1])

        a[1].loglog(epochs[1:],valid_perplexity[1:]- m2,"--",c='b')
        a[1].loglog(epochs[1:],valid_perplexity_2[:-1] -m2,c='r')
        a[1].set_ylabel("Test Perplexity (- {:0.1f})".format(m2),fontsize="x-large")
        a[1].set_xlabel("Iterations",fontsize="x-large")

        plb.subplots_adjust(left=.09, bottom=.1, right=.97, top=.97,
                            hspace=.1)


        plb.savefig("result_ptb.pdf", dpi=150)
        plb.show()
            
    
    
    
    
    

    

