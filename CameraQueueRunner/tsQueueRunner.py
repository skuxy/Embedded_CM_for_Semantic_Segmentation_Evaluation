import numpy as np
import tensorflow as tf
import model
import time
import cv2
import os
import sys
import helper
import pdb
import threading
from gi.repository import Aravis
from skimage import transform
class_names=['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation',
             'terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']


def getCameraStream():
    Aravis.enable_interface ("Fake")

    try:
        if len(sys.argv) > 1:
            camera = Aravis.Camera.new (sys.argv[1])
        else:
            camera = Aravis.Camera.new (None)
    except:
        print ("No camera found")
        exit()


    stream = camera.create_stream (None, None)

    #stream.push_buffer (Aravis.Buffer.new_allocate(payload))

    return stream, camera


stream, camera = getCameraStream()
[x,y,width,height] = camera.get_region()
camera.set_acquisition_mode(Aravis.AcquisitionMode.CONTINUOUS)
device = camera.get_device()
payload = camera.get_payload()
buffer = Aravis.Buffer.new_allocate(payload)
#device.set_string_feature_value("AcquisitionMode", "MultiFrame")'''
k = 4


'''
def printCameraData(camera):
    [x,y,width,height] = camera.get_region()
    print "Camera vendor : %s" %(camera.get_vendor_name ())
    print "Camera model  : %s" %(camera.get_model_name ())
    print "Camera id     : %s" %(camera.get_device_id ())
    print "ROI           : %dx%d at %d,%d" %(width, height, x, y)
    print "Payload       : %d" %(payload)
    print "Pixel format  : %s" %(camera.get_pixel_format_as_string ())
'''
init = False
timestamp_difference = 0
def nextFrame():    
    global buffer
    #print (time.time())
    stream.push_buffer(buffer)
    buffer = stream.pop_buffer()
    bufferTime = buffer.get_timestamp()
    if not init:
        global init, timestamp_difference
        timestamp_difference = int(round(time.time() * 1000000000)) - bufferTime
        print('TIMESTAMP DIFFERENCE: ', timestamp_difference)
        init = True
    data = buffer.get_data()
    #camera.stop_acquisition ()

    imgData = np.ndarray(buffer=data, dtype=np.uint8, shape=(height,width,1))
    img = cv2.cvtColor(imgData, cv2.COLOR_BAYER_RG2RGB)
    if img is None or img.size <= 0:
        raise ValueError('Image is None or empty')
    #print (img.shape)
    img = transform.resize(img, (256//k, 512//k,3), order=3)
    img = img.reshape((1,256//k, 512//k, 3))
    
    #print(img, bufferTime)
    return (img, bufferTime)



def evaluate(model, resume_path): 
    active = True
    camera.start_acquisition()
    with tf.Session() as sess:
        myShape = (1, 256//k, 512//k,3)
        #all_data = np.random.random(size=(30,1, 256//k,512//k, 3)).astype(dtype='float32')
        feature_input = tf.placeholder(tf.float32, shape=myShape)
        buffer_timing = tf.placeholder(tf.int64)

        queue = tf.FIFOQueue(1, dtypes=[tf.float32, tf.int64], shapes=[myShape, []])
        enqueue_op = queue.enqueue((feature_input, buffer_timing))
        inputs = queue.dequeue()

        sess.run(tf.global_variables_initializer())

        with tf.variable_scope('model'):
            logits, timestamp = model.build(inputs[0], inputs[1])

        print('\nRestoring params from:', resume_path)
        saver = tf.train.Saver(tf.global_variables(),sharded=False)
        saver.restore(sess, resume_path)
        sess.run(tf.local_variables_initializer())
	        

        def load_and_run():
            while active:
                sess.run(enqueue_op, feed_dict={(feature_input, buffer_timing): nextFrame()})
        
        t = threading.Thread(target=load_and_run)
        t.start()

        t0=time.time()
        for i in range(25):
            l, tmst = sess.run([logits, timestamp])
            curr = int(round(time.time() * 1000000000))
            print((curr-tmst-timestamp_difference)/1000000000)
            #pdb.set_trace()
            #print (i)
        #print(time.time()-t0)
        active = False
        time.sleep(1)
        sess.close()   
    camera.stop_acquisition()
   


def main(argv=None):
    model = helper.import_module('model', 'models/model_s.py')
    
    resume_path = 'trained_model/model.ckpt'

    evaluate(model, resume_path)


if __name__ == '__main__':
    main()

