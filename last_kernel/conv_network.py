import tensorflow as tf


def create_weight_variable(shape, collections=None, name=None):
    initial = tf.truncated_normal(shape, mean=0., stddev=0.1)
    collections += [tf.GraphKeys.VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
    return tf.Variable(
        initial_value=initial, collections=collections, name=name)


def create_bias_variable(shape, collections=None, name=None):
    initial = tf.constant(0.1, shape=shape)
    collections += [tf.GraphKeys.VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES]
    return tf.Variable(
        initial_value=initial, collections=collections, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


class ConvNetwork(object):
    """Convolutional Neural Nets"""
    def __init__(self, n_layers):
        super(ConvNetwork, self).__init__()
        self.n_layers = n_layers

        self._construct()
        self.reset()

    def reset(self):
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

    def _construct(self):
        self.graph = tf.Graph()

        with self.graph.as_default():

            # Data
            self.X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
            self.y_ = tf.placeholder(tf.float32, shape=[None, 10],
                                     name="label")
            self.lr = tf.constant(0.0001, dtype=tf.float32,
                                  name="learning_rate")
            keep_proba = tf.constant(1, dtype=tf.float32, name="keep_proba")
            self.mapping = {"X": self.X, "y": self.y_, "lr": self.lr,
                            "keep_proba": keep_proba}

            x_inp = tf.reshape(self.X, [-1, 28, 28, 1])

            # Create the network
            with tf.name_scope("Conv_1"):
                W_conv_1 = create_weight_variable(
                    [5, 5, 1, 32], collections=['kernel'], name="Wc_1")
                b_conv_1 = create_bias_variable(
                    [32], collections=['kernel'], name="bc_1")

                h_conv_1 = tf.nn.relu(conv2d(x_inp, W_conv_1) + b_conv_1)
                h_pool_1 = max_pool_2x2(h_conv_1)

            with tf.name_scope("Conv_2"):
                W_conv_2 = create_weight_variable(
                    [5, 5, 32, 64], collections=['kernel'], name="Wc_2")
                b_conv_2 = create_bias_variable(
                    [64], collections=['kernel'], name="bc_2")

                h_conv_2 = tf.nn.relu(conv2d(h_pool_1, W_conv_2) + b_conv_2)
                h_pool_2 = max_pool_2x2(h_conv_2)

            with tf.name_scope("Full_1"):
                W_fc_1 = create_weight_variable(
                    [7*7*64, 1024], collections=['kernel'], name="Wf_1")
                b_fc_1 = create_bias_variable(
                    [1024], collections=['kernel'], name="bf_1")

                h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])
                h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc_1) + b_fc_1)

                h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob=keep_proba)

            with tf.name_scope("Last_layer"):
                W_fc_2 = create_weight_variable(
                    [1024, 10], collections=['alpha'], name="Wf_2")
                b_fc_2 = create_bias_variable(
                    [10], collections=['alpha'], name="bf_2")

                # Network output
                self.output = tf.nn.softmax(
                    tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2)

            # Error
            self._cost = cross_entropy = tf.reduce_sum(
                -self.y_*tf.log(self.output))

            correct_prediction = tf.equal(tf.arg_max(self.output, 1),
                                          tf.arg_max(self.y_, 1))
            self._accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                                    tf.float32))

            # Training
            with tf.name_scope("Training"):
                self.optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lr)
                self._train_all = self.optimizer.minimize(loss=cross_entropy)
                self._train_alpha = self.optimizer.minimize(
                    loss=cross_entropy, var_list=tf.get_collection('alpha'))
                self._train_kernel = self.optimizer.minimize(
                    loss=cross_entropy, var_list=tf.get_collection('kernel'))

            # Initialization
            self.init = tf.initialize_all_variables()

            # Summary
            self.train_writer = tf.train.SummaryWriter('/tmp/lastKernel',
                                                       self.graph)

    def accuracy(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run(self._accuracy, feed_dict=feed)

    def train_all(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run([self._train_all, self._cost], feed_dict=feed)

    def train_alpha(self, **kwargs):
        feed = self._convert_feed(kwargs)
        return self.session.run([self._train_alpha, self._cost],
                                feed_dict=feed)

    def train_kernel(self, **kwargs):
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
