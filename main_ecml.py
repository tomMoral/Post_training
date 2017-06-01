# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:47:03 2016

@author: tomMoral
"""

try:
    import sys
    sys.path.remove("/usr/lib/python3/dist-packages")
except ValueError:
    pass
import os
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')
import pylab as plb
import numpy as np
import time
import copy
import pickle


# MNIST remated function

def print_mnist_image(image, label):
    image_to_print = image.reshape((28, 28))
    image_to_print = image_to_print[::-1, :]
    plb.pcolor(image_to_print, cmap='binary')
    title = np.argmax(label)
    plb.title(title)
    plb.show()


def print_mnist_conv_weight(weights):
    for i in range(32):
        image = weights[:, :, 0, i]
        plb.subplot(8, 4, i)
        plb.pcolor(image, cmap='binary')
        plb.title(i)
    plb.show()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('Experiments for last kernels')
    parser.add_argument('--gpu', type=float, default=.95,
                        help='ratio of the gpu usage allowed')
    parser.add_argument('--exp', type=str, default="classic",
                        help='ratio of the gpu usage allowed')

    args = parser.parse_args()

    N_step = 100
    N_alpha = 400

    from last_kernel.neural_network import NeuralNetwork
    from datasets_handler import cifar10_inputs

    exp_dir = os.path.join("exps", args.exp)
    archfile = os.path.join(exp_dir, "model.json")
    save_dir = os.path.join(exp_dir, "ckpt")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = None

    def save_results(file, obj):
        with open(os.path.join(exp_dir, file), "wb") as f:
            pickle.dump(obj, f)

    # Construct an input reader and a NeuralNet
    pb = cifar10_inputs.Cifar10Inputs()
    network = NeuralNetwork(archfile=archfile, pb=pb,
                            gpu_usage=args.gpu)

    # List containing (step, test_error_alpha, train_cost_alpha,
    #                  test_error_classic, train_cost_classic)
    curve = []
    # First, train the network for 100 iterations and collect
    true_error, saved_models, pretrain_log = network.train(
        max_steps=N_step+1, save_dir=save_dir, model_name="classic")
    model = copy.deepcopy(network.export_alpha_weights())
    for step in range(900):
        point = []
        # Train the network with last kernel
        t_start = time.time()
        network.import_alpha_weights(model)

        # Last kernel training of the model
        _err, _, _log = network.train(
            max_steps=N_alpha, alpha=True, save_dir=save_dir,
            model_name="alpha".format(step))
        point = list(_err[-1]) + [_log[-1][1][-1]]

        # Classic training of the model
        network.import_alpha_weights(model)
        _err, saved_models, _log = network.train(
            max_steps=N_step, alpha=False, save_dir=save_dir,
            model_name="classic".format(step))
        model = network.export_alpha_weights()

        point += [_err[-1][1]] + [_log[-1][1][-1]]
        curve += [point]

        # Periodically save the results to disk
        if step % 10 == 0:
            save_results("results.pkl", curve)
            print("Dump results on disk!!")

    save_results("results.pkl", curve)

    import IPython
    IPython.embed()

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[10.0, 6.0])
        err = np.array(curve)
        ax1.semilogy(err[:, 0], err[:, 1], label="LastKernel")
        ax1.semilogy(err[:, 0], err[:, 3], dashes=[6, 2], label="SGD")
        # ax1.set_ylim((.13, .3))
        ax1.legend()
        # ax1.set_xticks(np.linspace(0, 120000, 7))
        # ax1.set_yticks([.13, .15, .2, .3])
        # ax1.set_yticklabels([.13, '', .2])
        ax1.set_ylabel("Classification Error", fontsize="x-large")

        ax2.semilogy(err[:, 0], err[:, 2])
        ax2.semilogy(err[:, 0], err[:, 4], dashes=[6, 2])

        # ax2.set_ylim(.2, 2)
        # ax2.set_xticks(np.linspace(0, 120000, 7))
        # ax2.set_xticklabels(
        #     ['{}k'.format(k * 20) if k > 0 else 0 for k in range(7)])
        # ax2.set_yticks([.2, .5, 1])
        # ax2.set_yticklabels([.2, .5, 1])
        plt.xlabel("Iteration $T$", fontsize="x-large")
        ax2.set_ylabel("Training Cost", fontsize="x-large")

        plt.subplots_adjust(left=.09, bottom=.1, right=.97, top=.97)
        plt.savefig(os.path.join(exp_dir, "cifar10_full_ecml.pdf"), dpi=150)
        plt.show()
    except:
        pass

    import IPython
    IPython.embed()

    # Close network session
    network.session.close()
