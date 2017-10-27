# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:47:03 2016

@authors: audiffren, tommoral
"""
from __future__ import print_function

import os
import sys
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import linalg_ops as ops
from datasets_handler.parkinson_updrs import ParkinsonUPDRSInputs
from datasets_handler.simulated_regression import SimulatedRegressionInputs
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
font = {'family': 'serif',
        'weight': 'bold',
        'size': 18}

matplotlib.rc('font', **font)
matplotlib.rc('text', usetex=True)
import pylab as plb


SAVE_DIR = "save_exp"
if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# Parse program arguments
parser = argparse.ArgumentParser(description='Xp deep regression last kernel')

parser.add_argument('--xp_dim', metavar='xp_dim', type=int, default=1,
                    help='Number of the XP')
parser.add_argument('--reload', action='store_true',
                    help='Load the saved data and plot the figure')
parser.add_argument('--tab', action='store_true',
                    help='Output latex tabular row')
parser.add_argument('--simulated', action='store_true',
                    help='Use the simulated regression dataset instead of the '
                         'Parkinson Updrs dataset')
args = parser.parse_args()

xp_dim = args.xp_dim


load_figure = False

epochs_last_kernel = 200

save_error_every = 50
save_true_error_every = 50

batch_size = 50
if args.simulated:
    learning_rate = .05
    lmbd = 1e-3
    epochs = 751
    trajectory_starting = [250, 500, 750]
else:
    learning_rate = .01
    lmbd = 1e-3
    epochs = 751
    trajectory_starting = [250, 500, 750]


dic_var_save = {'batch_size': batch_size,
                'simulated': False,
                'epochs': epochs,
                'cp dim': xp_dim,
                'save_error_every': save_error_every,
                'save_true_error_every': save_true_error_every,
                'lmbd': lmbd}

fname = "regression_{}.pkl".format(
    "simulated" if args.simulated else "parkinson")
fname = os.path.join(SAVE_DIR, fname)

formatter = "Iteration {}: {:.3f}, {:.3f}, {:.3f}"
if args.tab:
    formatter = "{} & {:.3f} & {:.3f} & {:.3f} \\\\"

tab_error = []
if args.reload:
    with open(fname, "rb") as f:
        tab_error = pickle.load(f)
    for row in tab_error:
        print(formatter.format(*row))
else:

    # Random seed
    np.random.seed(0)
    tf.set_random_seed(1)

    # Load the dataset
    if args.simulated:
        dataset = SimulatedRegressionInputs(split_ratio=.8,
                                            batch_size=batch_size)
    else:
        dataset = ParkinsonUPDRSInputs(xp_dim=args.xp_dim, split_ratio=.8,
                                       batch_size=batch_size)
    ndim = dataset.n_dim
    start_time = time.time()


    # SHORTCUT functions
    def create_weight_variable(shape):
        initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
        return tf.Variable(initial_value=initial)

    def create_bias_variable(shape, trainable=True):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial_value=initial, trainable=trainable)

    # Data
    X = tf.placeholder(tf.float32, shape=[None, ndim])
    y_ = tf.placeholder(tf.float32, shape=[None, 1])
    lmbd_ = tf.constant(lmbd, dtype=tf.float32)

    keep_proba = tf.placeholder(tf.float32)

    # Create the network with 3 linear layers
    W_1 = create_weight_variable([ndim, ndim])
    b_1 = create_bias_variable([ndim])

    h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)
    h_1_drop = tf.nn.dropout(h_1, keep_prob=keep_proba)

    W_2 = create_weight_variable([ndim, 10])
    b_2 = create_bias_variable([10])

    h_2 = tf.nn.relu(tf.matmul(h_1_drop, W_2) + b_2)
    h_2_drop = tf.nn.dropout(h_2, keep_prob=keep_proba)

    last_layer = h_2_drop  # N * 10
    # shape of kernel matrix : N * N
    last_matrix = tf.matmul(last_layer, last_layer, transpose_b=True)

    # Optimal weights for last layer
    alpha = ops.matrix_solve(
        last_matrix + lmbd_ * dataset.N_train * np.identity(dataset.N_train),
        y_)
    final_weights = tf.matmul(last_layer, alpha, transpose_a=True)  # 10 * 1

    W_3 = create_weight_variable([10, 1])

    y = tf.matmul(h_2_drop, W_3)

    var_list = [W_1, W_2, W_3, b_1, b_2]

    reg_norm2 = 0
    for v in var_list:
        reg_norm2 += tf.reduce_sum(tf.square(v))

    # Error loss for the network
    krr = tf.reduce_mean(tf.square(y_ - y)) + lmbd_ * reg_norm2
    error_krr = tf.reduce_mean(tf.square(y_ - y))

    # Training
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss=krr)
    post_train_step = tf.train.GradientDescentOptimizer(
        learning_rate=learning_rate).minimize(loss=krr, var_list=[W_3])

    errors = []
    true_errors = []

    error_basic = []
    true_error_basic = []

    init = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init)

    total_x, total_y = dataset.get_train_set()
    x_test, y_test = dataset.get_test_set()

    total_feed_dict = {X: total_x, y_: total_y, keep_proba: 1.}
    test_feed_dict = {X: x_test, y_: y_test, keep_proba: 1.}

    value_list = []

    # Algorithm ---- First training step
    for batch in range(epochs):
        sys.stdout.write("\rpre-train: {:7.2%}".format(batch / epochs))
        sys.stdout.flush()
        batch_x, batch_y = dataset.get_next_batch()

        # Save starting points for post-training and optimal layer
        if (batch in trajectory_starting or batch + 1 == epochs):
            value_list.append((batch, [v.eval(session) for v in var_list]))

        # Classic training step
        session.run(train_step, feed_dict={X: batch_x, y_: batch_y,
                                           keep_proba: 1.})
    print("\rpre-train: done   ")

    # Perform post training with different starting points
    for (iteration_init, value_l) in value_list:

        # Re-load the weights for the begining of the trajectory
        for i, v in enumerate(var_list):
            session.run(v.assign(value_l[i]))

        # Compute error before post-training
        train_cost_init = session.run(krr, feed_dict=total_feed_dict)
        test_error_init = session.run(error_krr, feed_dict=test_feed_dict)

        # Run epochs_post-training steps with post_training
        for batch in range(epochs_last_kernel):
            batch_x, batch_y = dataset.get_next_batch()
            session.run(post_train_step, feed_dict={X: batch_x, y_: batch_y,
                                                    keep_proba: 1.})

        # Compute error after post-training
        train_cost_pt = session.run(krr, feed_dict=total_feed_dict)
        test_error_pt = session.run(error_krr, feed_dict=test_feed_dict)

        # Algorithm ---- Exact Solution

        # Re-load the weights for the begining of the trajectory
        for i, v in enumerate(var_list):
            session.run(v.assign(value_l[i]))

        # DEBUG
        test_error_init2 = session.run(error_krr, feed_dict=test_feed_dict)
        assert test_error_init2 == test_error_init

        # Compute the optimal value of the last layer with close form solution
        fd = final_weights.eval(session=session, feed_dict={
            X: total_x, y_: total_y, keep_proba: 1.}) 
        session.run(W_3.assign(fd))

        # Compute train/test errors and store it
        train_cost_opt = session.run(krr, feed_dict=total_feed_dict)
        test_error_opt = session.run(error_krr, feed_dict=test_feed_dict)

        tab_error += [(iteration_init, test_error_init, test_error_pt,
                       test_error_opt)]
        print("Iteration {}: {:.3f}, {:.3f}, {:.3f}".format(*tab_error[-1]))

    session.close()
    print("All done, time elapsed", time.time() - start_time)

    with open(fname, "wb") as f:
        pickle.dump(tab_error, f)
