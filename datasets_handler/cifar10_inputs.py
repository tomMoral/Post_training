import os
import sys
import numpy as np
import tensorflow as tf

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_DIR = os.path.join("data", "Cifar10")

CIFAR_DIR = os.path.join(DATA_DIR, 'cifar-10-batches-bin')

IM_SIZE = 24
NUM_PREPROCESS_THREADS = 16

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


class Cifar10Inputs(object):
    """A simple problem to test the capability of LISTA

    Parameters
    ----------
    D: array-like, [K, p]
        dictionary for the generation of problems. The shape should be
        [K, p] where K is the number of atoms and p the dimension of the
        output space
    lmbd: float
        sparsity factor for the computation of the cost
    """
    def __init__(self, batch_size=128, data_dir=DATA_DIR, seed=None):
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed

        # Load training images if needed
        maybe_download_and_extract()

    @staticmethod
    def _read_cifar10(filename_queue):
        """Reads and parses examples from CIFAR10 data files.
        Recommendation: if you want N-way read parallelism, call this function
        N times.  This will give you N independent Readers reading different
        files & positions within those files, which will give better mixing of
        examples.
        Args:
            filename_queue: A queue of strings with the filenames to read from.
        Returns:
            An object representing a single example, with the following fields:
            height: number of rows in the result (32)
            width: number of columns in the result (32)
            depth: number of color channels in the result (3)
            key: a scalar string Tensor describing the filename & record number
                for this example.
            label: an int32 Tensor with the label in the range 0..9.
            uint8image: a [height, width, depth] uint8 Tensor with the data
        """

        class CIFAR10Record(object):
            pass
        result = CIFAR10Record()

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
        # the input format.
        label_bytes = 1  # 2 for CIFAR-100
        result.height = 32
        result.width = 32
        result.depth = 3
        image_bytes = result.height * result.width * result.depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        # Read a record, getting filenames from the filename_queue.  No
        # header or footer in the CIFAR-10 format, so we leave header_bytes
        # and footer_bytes at their default of 0.
        reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
        result.key, value = reader.read(filename_queue)

        # Convert from a string to a vector of uint8 that is record_bytes long.
        record_bytes = tf.decode_raw(value, tf.uint8)

        # The first bytes represent the label, which we convert from
        # uint8->int32.
        result.label = tf.cast(
            tf.slice(record_bytes, [0], [label_bytes]), tf.int32)

        # The remaining bytes after the label represent the image, which we
        # reshape from [depth * height * width] to [depth, height, width].
        depth_major = tf.reshape(
            tf.slice(record_bytes, [label_bytes], [image_bytes]),
            [result.depth, result.height, result.width])
        # Convert from [depth, height, width] to [height, width, depth].
        result.uint8image = tf.transpose(depth_major, [1, 2, 0])

        return result

    def _generate_image_and_label_batch(self, image, label, min_queue_examples,
                                        shuffle):
        """Construct a queued batch of images and labels.
        Args:
            image: 3-D Tensor of [height, width, 3] of type.float32.
            label: 1-D Tensor of type.int32
            min_queue_examples: int32, minimum number of samples to retain
                in the queue that provides of batches of examples.
            batch_size: Number of images per batch.
            shuffle: boolean indicating whether to use a shuffling queue.
        Returns:
            images: Images. 4D tensor of [batch_size, height, width, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.
        if shuffle:
            images, label_batch = tf.train.shuffle_batch(
                [image, label], batch_size=self.batch_size,
                num_threads=NUM_PREPROCESS_THREADS,
                capacity=min_queue_examples + 3*self.batch_size,
                min_after_dequeue=min_queue_examples,
                seed=self.seed)
        else:
            images, label_batch = tf.train.batch(
                [image, label], batch_size=self.batch_size,
                num_threads=NUM_PREPROCESS_THREADS,
                capacity=min_queue_examples + 3*self.batch_size)
        # Display the training images in the visualizer.
        return images, tf.reshape(label_batch, [self.batch_size])

    def get_train_inputs(self, distorted=False):
        return self._get_inputs(eval_data=False, distorted=distorted)

    def get_test_inputs(self):
        return self._get_inputs(eval_data=True, distorted=False)

    def _get_inputs(self, eval_data, distorted):
        """Construct input for CIFAR evaluation using the Reader ops.
        Args:
            eval_data: bool.
                Indicating if one should use the train or eval data.
        Returns:
            images: Images. 4D tensor. [batch_size, IM_SIZE, IM_SIZE, 3]
            labels: Labels. 1D tensor. [batch_size]
        """
        if not eval_data:
            filenames = [os.path.join(CIFAR_DIR, 'data_batch_%d.bin' % i)
                         for i in range(1, 6)]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
            shuffle = True
        else:
            filenames = [os.path.join(CIFAR_DIR, 'test_batch.bin')]
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
            shuffle = False

        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Create a queue that produces the filenames to read.
        filename_queue = tf.train.string_input_producer(filenames)

        # Read examples from files in the filename queue.
        read_input = self._read_cifar10(filename_queue)
        image = tf.cast(read_input.uint8image, tf.float32)

        # Image processing for evaluation.

        if distorted:
            # Randomly crop a [height, width] section of the image.
            image = tf.random_crop(image, [IM_SIZE, IM_SIZE, 3])
            # Because these operations are not commutative, consider
            # randomizing the order their operation.
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image,  max_delta=63)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        else:
            # Crop the central [height, width] of the image.
            image = tf.image.resize_image_with_crop_or_pad(
                image, IM_SIZE, IM_SIZE)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(image)

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *
                                 min_fraction_of_examples_in_queue)

        # Generate a batch of images and labels by building up a queue of
        # examples.
        return self._generate_image_and_label_batch(
            float_image, read_input.label, min_queue_examples, shuffle=shuffle)


def maybe_download_and_extract():
    """Download and extract the tarball from Alex's website."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        import urllib.request
        import tarfile

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading {} {:7.2%}'
                             .format(filename, count*block_size/total_size))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print('\r>> Downloading {} {:7}'.format(filename, 'done'))
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(DATA_DIR)
