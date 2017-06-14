import numpy as np
import tensorflow as tf
import model
import time
import os
import sys
import helper
import ipdb
from PIL import Image

class_names=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation',
             'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']

def collect_confusion(logits, labels, conf_mat):
    predicted_labels = logits.argmax(3).astype(np.int32, copy=False)

    num_examples = FLAGS.batch_size * FLAGS.img_height * FLAGS.img_width
    predicted_labels = np.resize(predicted_labels, [num_examples, ])
    batch_labels = np.resize(labels, [num_examples, ])

    assert predicted_labels.dtype == batch_labels.dtype
    eval_helper.collect_confusion_matrix(predicted_labels, batch_labels, conf_mat, FLAGS.num_classes)



def evaluate(sess, epoch_num, labels, logits, loss, data_size,name):
    print('\nTest performance:')
    loss_avg = 0

    batch_size = FLAGS.batch_size
    print('testsize = ', data_size)
    assert data_size % batch_size == 0
    num_batches = data_size // batch_size
    start_time=time.time()

    conf_mat = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.uint64)
    for step in range(1,num_batches+1):
        if(step%5==0):
            print('evaluation: batch %d / %d '%(step,num_batches))

        out_logits, loss_value, batch_labels = sess.run([logits, loss, labels])

        loss_avg += loss_value
        collect_confusion(out_logits,batch_labels,conf_mat)


    print('Evaluation {} in epoch {} lasted {}'.format(name,epoch_num,train_helper.get_expired_time(start_time)))

    print('')

    (acc, iou, prec, rec, _) = eval_helper.compute_errors(conf_mat,name,class_names,verbose=True)
    loss_avg /= num_batches;

    print('IoU=%.2f Acc=%.2f Prec=%.2f Rec=%.2f' % (iou,acc, prec, rec))
    return acc,iou, prec, rec, loss_avg


def evalone(model, resume_path, image):
    # define inputs, labels, weights
    #with tf.Graph().as_default():
    #sess = tf.Session()
    k=8
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.variable_scope('model'):
            #logits, loss = model.build(image, labels, weights)
            img = tf.placeholder(tf.float32, shape=[1, 256//k, 512//k, 3])
            logits = model.build(img)
        print('\nRestoring params from:', resume_path)
        saver = tf.train.Saver(tf.global_variables(),sharded=False)
        #saver = tf.train.Saver()
        saver.restore(sess, resume_path)
        sess.run(tf.local_variables_initializer())

        #out_logits, loss_value, batch_labels = sess.run([logits, loss, labels])
        #ipdb.set_trace()
        t0=time.time()
        out_logits = sess.run(logits, feed_dict={img: np.zeros([1, 256//k, 512//k, 3])})
        print(time.time()-t0)
        sess.close()
        print(out_logits.shape)


def main(argv=None):
    #ipdb.set_trace()
    # import model
    model = helper.import_module('model', 'models/model_s.py')
    
    # load image
    '''
    filename_queue = tf.train.string_input_producer(['rgb_img.png'])
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_png(value)    
    print (tf.shape(image))
    exit()
    
    image = Image.open('rgb_img.png')
    image = np.asarray(image)
    image = image.astype('uint8')    
    '''
    filename_queue = tf.train.string_input_producer(['rgb_img.png'])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'num_labels': tf.FixedLenFeature([], tf.int64),
            'img_name': tf.FixedLenFeature([], tf.string),
            'rgb': tf.FixedLenFeature([], tf.string),
            'label_weights': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),

        })

    image = tf.decode_raw(features['rgb'], tf.uint8)

    image = tf.reshape(image, shape=[1, 256, 512, 3])
    image=tf.to_float(image)
    
    # define path where model checkpoint is stored
    resume_path = '/home/stjepan/FAKS/zavrad/VGG_TX1/model.ckpt' 
    
    # evaluate one image
    evalone(model, resume_path, image)


if __name__ == '__main__':
    main()
