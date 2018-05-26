# import train_helper
# import time
# import os
import helper
# import sys

# import eval_helper
import numpy as np
import tensorflow as tf
import read_cityscapes_tf_records as reader

tf.app.flags.DEFINE_string('config_path', "config/cityscapes.py",
                           """Path to experiment config.""")
FLAGS = tf.app.flags.FLAGS

helper.import_module('config', FLAGS.config_path)
print(FLAGS.__dict__['__flags'].keys())


def main(argv=None):

    train_data, train_labels, train_names, train_weights = reader.inputs(
        shuffle=True, num_epochs=1, dataset_partition='train')
    session = tf.Session()
    session.run(tf.initialize_local_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(session=session, coord=coord)

    for i in range(1):  # WAS
        print(i)

        labels, weights = session.run([train_labels, train_weights])
        l255 = labels[0, labels[0] == 255]

        result_sum = 0
        for j in range(19):
            print('Label {}'.format(j))

            lj = labels[0, labels[0] == j]
            wj = weights[0, labels[0] == j]
            amount = len(lj) / (len(labels[0]))

            print(amount)

            if len(wj) > 0:
                print('Weight ', wj[0])
                d = wj[0] * amount

            else:
                d = 0
            result_sum += d
        print(result_sum)

    coord.request_stop()
    coord.join(threads)
    session.close()


if __name__ == '__main__':
    tf.app.run()
