"""
Created on Tue Apr 12 14:47:03 2016

@author: tomMoral
"""
import os
import math
import time
import logging
import numpy as np
import tensorflow as tf
from .utils import start_handler

DTYPE = tf.float32

N_SAMPLE_PER_ECPOCH = 50000
# N_EPOCH_PER_DECAY = 100
N_EPOCH_PER_DECAY = 10
MOVING_AVERAGE_DECAY = .9999


class NeuralNetwork(object):
    """Neural Nets object"""
    def __init__(self, archfile, pb=None, seed=None, gpu_usage=1):

        import json
        with open(archfile, 'r') as f:
            conf_file = json.load(f)

        self.pb = pb
        self.mapping = {}
        self.gpu_usage = gpu_usage
        self._config = conf_file['network']
        self._training_conf = conf_file['training']

        self._logger = logging.getLogger("NeuralNet")
        start_handler(self._logger, 10)
        self._construct()
        self.reset()

    def reset(self):
        """Reset the state of the network."""
        if hasattr(self, 'session'):
            self.session.close()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_usage
        self.session = tf.Session(graph=self.graph, config=config)
        self.session.run(self._init)
        tf.train.start_queue_runners(sess=self.session)

    def _linear_layer(self, inputs, shape, name, activation=None, reg_l2=0,
                      biases=None, stddev=5e-2, conv=False, collection=None,
                      reg_l2_alpha=None, **layer_conf):

        with tf.variable_scope(name):
            if biases is not None:
                bias_shape = shape[-1]
                hk = tf.get_variable(
                    name='biases', shape=bias_shape, dtype=DTYPE,
                    initializer=tf.constant_initializer(biases))
                if collection is not None:
                    tf.add_to_collection(collection, hk)
            if conv:
                init = tf.truncated_normal_initializer(stddev=stddev,
                                                       dtype=DTYPE)
                W = tf.get_variable("weights", shape=shape, initializer=init)
                hk = tf.add(tf.nn.conv2d(inputs, W, **layer_conf), hk,
                            name=name)
            else:
                assert len(layer_conf) == 0, (
                    "unknown kwargs {}".format(layer_conf))
                shape = [inputs.get_shape()[1].value]+shape
                self._logger.debug("linear layer with shape {}".format(shape))
                init = tf.truncated_normal_initializer(stddev=stddev,
                                                       dtype=DTYPE)
                W = tf.get_variable("weights", shape=shape, initializer=init)
                hk = tf.add(tf.matmul(inputs, W), hk, name=name)
            if collection is not None:
                tf.add_to_collection(collection, W)
                if collection == "alpha":
                    if reg_l2_alpha is None:
                        reg_l2_alpha = reg_l2
                    if reg_l2_alpha > 0:
                        lmbd = tf.constant(reg_l2_alpha)
                        self.mapping["lmbd"] = lmbd
                        weight_loss = tf.multiply(lmbd, tf.nn.l2_loss(W),
                                                  name='weight_loss_alpha')
                        tf.add_to_collection('losses_alpha', weight_loss)
            if reg_l2 > 0:
                weight_loss = tf.multiply(reg_l2, tf.nn.l2_loss(W),
                                          name='weight_loss')
                tf.add_to_collection('losses', weight_loss)
            if activation == 'relu':
                hk = tf.nn.relu(hk, name=name)
            return hk

    def _mk_layer(self, layer_conf, inputs, collection):
        l_type = layer_conf.pop('type')
        self._logger.debug("constructing layer {} ({})"
                           .format(layer_conf['name'], l_type))
        if 'linear' in l_type:
            return self._linear_layer(inputs, conv='conv' in l_type,
                                      collection=collection, **layer_conf)
        elif l_type == "max_pool":
            return tf.nn.max_pool(inputs, **layer_conf)
        elif l_type == "lrn":
            return tf.nn.lrn(inputs, **layer_conf)
        elif l_type == "flatten":
            return tf.contrib.layers.flatten(inputs)
        elif l_type == "dropout":
            keep_prob = tf.constant(1., tf.float32)
            key = "keep_prob_{}".format(layer_conf['name'])
            self.mapping[key] = keep_prob
            return tf.nn.dropout(inputs, keep_prob)
        else:
            raise RuntimeError("unknown layer type {}".format(l_type))

    def _get_inputs(self):
        """Create the input placeholders, to feed as first layer of the network

        If a datahandler is fournished, create a switch to either fetch the
        training or testing data.
        """

        if self.pb is not None and hasattr(self.pb, "_get_inputs"):
            with tf.name_scope("inputs_readers"):
                data, labels = self.pb.get_train_inputs(distorted=True)
                data_val, labels_val = self.pb.get_test_inputs()
            eval_data = tf.placeholder_with_default(
                tf.constant(False), shape=(), name="eval_data")

            # If `eval_data` is set to True in the feed dictionary, use the
            # test set to perform the operation.
            self.mapping["eval_data"] = eval_data
            self.X = tf.cond(eval_data, lambda: data_val, lambda: data)
            self.y_ = tf.cond(eval_data, lambda: labels_val, lambda: labels)
        else:
            inputs_conf = self._config['inputs']
            input_shape = [None if v == -1 else v
                           for v in inputs_conf['shape']]
            label_shape = self._config['outputs']
            label_shape = [None if v == -1 else v for v in label_shape]
            self.X = tf.placeholder(DTYPE, shape=input_shape, name="X")
            self.y_ = tf.placeholder(DTYPE, shape=label_shape, name="label")

        self.mapping["X"] = self.X
        self.mapping["y"] = self.y_
        return self.X

    def _construct(self):
        self.graph = tf.Graph()

        with self.graph.as_default():

            output = self._get_inputs()

            # Define placeholder for training control
            self.global_step = tf.Variable(0.0, trainable=False)

            # Construct the graph
            collection = 'kernel'
            for i, layer_conf in enumerate(self._config['layers']):
                if i == len(self._config['layers'])-1:
                    collection = 'alpha'
                output = self._mk_layer(layer_conf, output, collection)
            self._output = output

            # Construct cost function
            self._cost, self._cost_alpha = self._get_cost(output)

            # Training
            self._mk_train_steps()

            tf.summary.scalar("accuracy", self._accuracy)

            # Initialization
            self._init = tf.initialize_all_variables()

            # Summary
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver(tf.all_variables())
            self.saver_hard = tf.train.Saver(tf.all_variables(),
                                             max_to_keep=2)

            # Store the variable list to permit export/import
            self._alpha_vars = list(tf.get_collection('alpha') +
                                    [self.global_step])

    def _mk_train_steps(self):
        """construct op for training based on the given cost and learning_rate
        """
        step = self.global_step

        # Learning rate definition
        n_batches_per_epoch = N_SAMPLE_PER_ECPOCH / self.pb.batch_size
        decay_steps = int(N_EPOCH_PER_DECAY*n_batches_per_epoch)
        scale_lr = tf.constant(1.)
        lr = scale_lr*tf.train.exponential_decay(
            learning_rate=.1, global_step=self.global_step,
            decay_steps=decay_steps, decay_rate=.1, staircase=True)
        tf.summary.scalar("learning_rate", lr)
        self.mapping['lr'] = lr
        self.mapping['scale_lr'] = scale_lr

        loss_averages_op = self._add_loss_summaries(self._cost)

        with tf.control_dependencies([loss_averages_op]):
            self._opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
            grads = self._opt.compute_gradients(loss=self._cost)
            grads_alpha = self._opt.compute_gradients(
                loss=self._cost_alpha, var_list=tf.get_collection('alpha'))
        var_kernel = tf.get_collection('kernel')
        grads_kernel = [(g, v) for g, v in grads if v in var_kernel]
        apply_all = self._opt.apply_gradients(grads, global_step=step)
        apply_alpha = self._opt.apply_gradients(grads_alpha, global_step=step)
        apply_kernel = self._opt.apply_gradients(
            grads_kernel, global_step=step)

        # Track the moving averages of all trainable variables.
        average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, step)
        averages_var_alpha = average.apply(tf.get_collection('alpha'))
        averages_var_kernel = average.apply(var_kernel)

        # track gradients norms
        self.nl2_alpha = tf.add_n([tf.nn.l2_loss(g) for g, _ in grads_alpha
                                   if g is not None])
        self.nl2_full = tf.add_n([tf.nn.l2_loss(g) for g, _ in grads
                                  if g is not None])

        with tf.control_dependencies([apply_all, averages_var_alpha,
                                      averages_var_kernel]):
            self._train_all = tf.no_op(name='train_all')
        with tf.control_dependencies([apply_alpha, averages_var_alpha]):
            self._train_alpha = tf.no_op(name='train_alpha')
        with tf.control_dependencies([apply_kernel, averages_var_kernel]):
            self._train_kernel = tf.no_op(name='train_kernel')

    def _get_cost(self, output):
        # Construct cost function
        self._logger.debug("constructing cost function (softmax)")
        labels = tf.cast(self.y_, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=output, labels=labels, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        tf.add_to_collection('losses_alpha', cross_entropy_mean)

        correct_prediction = tf.equal(tf.arg_max(output, 1),
                                      labels)
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                tf.float32))

        self._top_k = tf.nn.in_top_k(output, labels, 1)

        return (tf.add_n(tf.get_collection('losses'), name='total_loss'),
                tf.add_n(tf.get_collection('losses_alpha'), name='alpha_loss'))

    def _add_loss_summaries(self, total_loss):
        """Add summaries for losses in CIFAR-10 model.

        Generates moving average for all losses and associated summaries for
        visualizing the performance of the network.

        Args:
            total_loss: Total loss from loss().
        Returns:
            loss_averages_op: op for generating moving averages of losses.
        """
        # Compute the moving average of all individual losses and the total
        # loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply([total_loss])

        # Attach a scalar summary to the total loss.
        tf.summary.scalar(total_loss.op.name, total_loss)

        return loss_averages_op

    def accuracy(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run(self._accuracy, feed_dict=feed)

    def top_k(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run(self._top_k, feed_dict=feed)

    def train_step_all(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run([self._train_all, self._cost], feed_dict=feed)

    def train_step_alpha(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run([self._train_alpha, self._cost],
                                feed_dict=feed)

    def train_step_kernel(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run([self._train_kernel, self._cost],
                                feed_dict=feed)

    def _convert_feed(self, kwargs):
        feed = {}
        for k, v in kwargs.items():
            try:
                feed[self.mapping[k]] = v
            except KeyError:
                raise AttributeError("No parameter {} in this network"
                                     .format(k))
        return feed

    def train(self, max_steps=1000, train_dir="/tmp/neural_net", alpha=False,
              save_dir=None, save_step=10000, model_name="network",
              scale_lr=1, **train_conf):
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        model_name += "_alpha" if alpha else ""
        train_dir = os.path.join(train_dir, model_name)
        if os.path.exists(train_dir):
            import shutil
            shutil.rmtree(train_dir)
        os.mkdir(train_dir)

        class FakeWriter():
            def add_summary(self, a, global_step=0):
                pass
        # train_writer = tf.summary.FileWriter(train_dir, self.graph)
        train_writer = FakeWriter()
        self.eval_once()

        tst_error, saved_models, log = [], [], []
        log = []
        if alpha:
            trn_conf = self._training_conf.get('alpha', {})
            step_func = self.train_step_alpha
        else:
            trn_conf = self._training_conf.get('classic', {})
            step_func = self.train_step_all
        trn_conf.update(train_conf)
        for step in range(max_steps):

            # Perform the training step
            start_time = time.time()
            _, loss_value = step_func(**trn_conf)
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0 or (step + 1) == max_steps:
                global_step = self.global_step.eval(session=self.session)
                examples_per_sec = self.pb.batch_size / duration
                sec_per_batch = float(duration)

                format_str = ("step {:.0f}, loss = {} ({:.1f} examples/sec; "
                              "{:.3f} sec/batch)")
                print(format_str.format(global_step, loss_value,
                                        examples_per_sec, sec_per_batch))

            if step > 0 and (step % 100 == 0 or (step + 1) == max_steps):
                acc, summary_str, global_step = self.session.run(
                    [self._accuracy, self.summary_op, self.global_step])
                train_writer.add_summary(summary_str, global_step=global_step)
                print("Step {:.0f}: Accuracy {:.4f}".format(global_step, acc))

            # print test error periodically.
            if step > 0 and (step % 500 == 0 or (step + 1) == max_steps):
                global_step = self.global_step.eval(session=self.session)
                acc = self.eval_once()
                tst_error += [(global_step, 1 - acc)]
                # checkpoint_path = os.path.join(train_dir, 'network.ckpt')
                # self.saver.save(self.session, checkpoint_path,
                #                 global_step=self.global_step)
                ng, nga, loss_value = self.session.run([
                    self.nl2_full, self.nl2_alpha, self._cost])
                log += [(global_step, (ng, nga, loss_value))]

            if (save_dir and (step % save_step == 0 or (step + 1) == max_steps)
                    and step > 0):
                checkpoint_path = os.path.join(save_dir, '{}.ckpt'.format(
                    model_name))
                saved_models += [(step, self.save(checkpoint_path))]

        return tst_error, saved_models, log

    def save(self, checkpoint_path):
        print("saved to ", checkpoint_path)
        return self.saver_hard.save(self.session, checkpoint_path,
                                    global_step=self.global_step)

    def restore(self, checkpoint_path):
        print("restore ", checkpoint_path)
        return self.saver_hard.restore(self.session, checkpoint_path)

    def export_alpha_weights(self):
        return self.session.run(self._alpha_vars)

    def import_alpha_weights(self, weights):
        assert len(weights) == len(self._alpha_vars),\
            "Wrong number of variables"
        with self.session.as_default():
            import_op = [V.assign(v) for V, v in zip(self._alpha_vars,
                                                     weights)]
            self.session.run(import_op)
            global_step = self.global_step.eval()
            print("Network restored to state {:.0f}".format(global_step))

    def eval_once(self):
        """Run Eval once.
        Args:
            saver: Saver.
            summary_writer: Summary writer.
            top_k_op: Top K op.
            summary_op: Summary op.
        """
        sess = self.session
        start_time = time.time()

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(
                    sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(10000 / self.pb.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * self.pb.batch_size
            step = 0
            feed = {self.mapping['eval_data']: True}
            while step < num_iter and not coord.should_stop():
                predictions = sess.run(self._top_k, feed_dict=feed)
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1
            precision = true_count / total_sample_count

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            precision = 0

        duration = time.time() - start_time
        examples_per_sec = total_sample_count / duration
        sec_per_batch = duration / num_iter

        format_str = ("Test accurracy = {:.3%} ({:.1f} examples/sec; "
                      "{:.3f} sec/batch)")
        print("=" * 50)
        print(format_str.format(precision, examples_per_sec,
                                sec_per_batch))
        print("=" * 50)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
        return precision
