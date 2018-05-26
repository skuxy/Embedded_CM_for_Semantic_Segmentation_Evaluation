import os
import pickle
import numpy as np
import tensorflow as tf
import skimage as ski
import skimage.data
import skimage.transform
from tqdm import trange
import math
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imresize

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('cx_start', 0, '')
tf.app.flags.DEFINE_integer('cx_end', 2048, '')
tf.app.flags.DEFINE_integer('cy_start', 0, '')
tf.app.flags.DEFINE_integer('cy_end', 900, '')

tf.app.flags.DEFINE_integer('img_width', 512, '')
tf.app.flags.DEFINE_integer('img_height', 256, '')


def create_tfrecord(rgb, label_map, weight_map, depth_img, num_labels, img_name, save_dir):
    # rows = rgb.shape[0]
    # cols = rgb.shape[1]
    # depth = rgb.shape[2]
    filename = os.path.join(save_dir + img_name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    rgb_str = rgb.tostring()
    labels_str = label_map.tostring()
    weights_str = weight_map.tostring()
    # disp_raw = depth_img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'num_labels': _int64_feature(int(num_labels)),
        'img_name': _bytes_feature(img_name.encode()),
        'rgb': _bytes_feature(rgb_str),
        'label_weights': _bytes_feature(weights_str),
        'labels': _bytes_feature(labels_str)})
    )

    writer.write(example.SerializeToString())

    writer.close()


def resizeMostCommonClass(gt_ids, downscale = 2):
    cx_start = FLAGS.cx_start
    cx_end = FLAGS.cx_end
    cy_start = FLAGS.cy_start
    cy_end = FLAGS.cy_end
    gt_ids = np.ascontiguousarray(gt_ids[cy_start:cy_end,cx_start:cx_end])
    myNp = np.asarray([np.argmax(np.bincount(np.ravel(np.ascontiguousarray(gt_ids[i:i+downscale, j:j+downscale]))))
        for i in range(cy_start, cy_end, downscale) for j in range(cx_start, cx_end, downscale)])
    myNp = np.reshape(myNp, (FLAGS.img_height, FLAGS.img_width)).astype(np.uint8)
    return myNp


def nearest_neighbor(gt_ids):
    return imresize(gt_ids, size=( FLAGS.img_height, FLAGS.img_width), interp='nearest')


def main(argv):
    cx_start = FLAGS.cx_start
    cx_end = FLAGS.cx_end
    cy_start = FLAGS.cy_start
    cy_end = FLAGS.cy_end

    path_to_script = os.path.realpath(__file__)
    filtered_path = path_to_script.split('/')[:-1]
    filtered_path = "/".join(filtered_path)

    rgb_path = filtered_path + '/aachen_000015_000019.ppm'
    rgb = ski.data.load(rgb_path)
    rgb = np.ascontiguousarray(rgb[cy_start:cy_end,cx_start:cx_end,:])
    rgb = ski.transform.resize( rgb, (FLAGS.img_height, FLAGS.img_width), preserve_range=True, order=3)
    rgb = rgb.astype(np.uint8)
    ski.io.imsave(filtered_path + '/rgb_img_no_hauba.png', rgb)
    exit()
    rgb = ski.transform.pyramid_reduce(rgb, downscale=4.0, order = 3)
    rgb = ski.img_as_ubyte(rgb)
    #rgb = rgb.astype(np.uint8)
    print (rgb)
    ski.io.imsave(filtered_path + '/rgb_pyrReduce_img.png', rgb)

    exit()

    #gt_path = './aachen_000015_000019.pickle'
    gt_path = filtered_path + 'aachen_000015_000019.pickle'
    with open(gt_path, 'rb') as f:
        gt_data = pickle.load(f)
    gt_ids = gt_data[0]
    #ski.io.imsave('./mostCommonTry1.png', gt_ids)
    #ski.io.imsave('mostCommonTry2.png', gt_ids)
    #exit()

    #gt_weights = gt_data[1]
    num_labels = gt_data[2]
    #print(num_labels)
    class_weights = gt_data[4]
    assert num_labels == (gt_ids < 255).sum()
    gt_ids = np.ascontiguousarray(gt_ids[cy_start:cy_end,cx_start:    cx_end])
    #gt_ids = resizeMostCommonClass(gt_ids, 4).astype(np.uint8)
    #gt_ids = resizeClosestNeighbor(gt_ids,FLAGS.cx_end, FLAGS.cy_end, FLAGS.img_width, FLAGS.img_height ).astype(np.uint8)
    #gt_ids = ski.transform.resize(gt_ids, (FLAGS.img_height, FLAGS.img_width), order=0, preserve_range=True).astype(np.uint8)
    gt_ids = nearest_neighbor(gt_ids).astype(np.uint8)
    print(np.shape(gt_ids))
    ski.io.imsave(filtered_path + '/gt_closestNeighbor.png', gt_ids)
    #ski.io.imsave('./gt_img.png', gt_ids)
    '''gt_weights = np.zeros((FLAGS.img_height, FLAGS.img_width), np.float32)
    for i, wgt in enumerate(class_weights):
        gt_weights[gt_ids == i] = wgt
    create_tfrecord(rgb, gt_ids, gt_weights, depth_img, num_labels, 'eeeeee', '/home/stjepan/FAKS/zavrad/tTf/')'''

if __name__ == '__main__':
    tf.app.run()
    main()
