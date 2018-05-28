import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

path = "/media/hlee/Work/set/output.tfrecords"
files = tf.train.match_filenames_once(path)#获取所有符合正则表达式的文件,返回文件列表
filename_queue = tf.train.string_input_producer([path],shuffle=False)  # create a queue

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  # return file_name and file

features = tf.parse_single_example(serialized_example,
                                   features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img': tf.FixedLenFeature([], tf.string),
                                       'width': tf.FixedLenFeature([], tf.int64),
                                       'height': tf.FixedLenFeature([], tf.int64),
                                       'channels':tf.FixedLenFeature([],tf.int64)
                                   })  # return image and label

# img = test_tf.image.convert_image_dtype(img, dtype=test_tf.float32)
# img = test_tf.reshape(img, [512, 80, 3])  # reshape image to 512*80*3
# img = test_tf.cast(img, test_tf.float32) * (1. / 255) - 0.5  # throw img tensor

label = tf.cast(features['label'], tf.int32)  # throw label tensor
# height = tf.cast(features['height'],tf.int32)
height = features['height']
width = tf.cast(features['width'],tf.int32)
channels = tf.cast(features['channels'],tf.int32)


img = tf.decode_raw(features['img'], tf.uint8)
print(type(height))
img = tf.reshape(img,[227,227,3])

# img.set_shape([324,324,1])
# label = test_tf.reshape(label,[1])
# img = test_tf.image.resize_images(img,[width,height],method=1)

# label = test_tf.reshape(label,[1])
# img.set_shape([height,width,1])
# img.set_shape([height,width])
# img = test_tf.image.convert_image_dtype(img,dtype=test_tf.float32)
# img = test_tf.image.resize_images(img,[height,width],method=0)

# img=test_tf.cast(img,test_tf.float32)*(1./255)-0.5



#
batch_size = 3
min_after_dequeue = 10
capacity = 100 + 3 * batch_size
img_batch, label_batch = tf.train.shuffle_batch([img,label],
                                                batch_size=64,
                                                capacity=22000,
                                                min_after_dequeue=20000,
                                                num_threads=4)
#
label_batch = tf.reshape(label_batch, [64,1])
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # sess.run(test_tf.local_variables_initializer())#使用tf.train.match_filenames_once(path)需要这句
    sess.run(tf.global_variables_initializer())
    xs, ys = sess.run([img_batch,label_batch])
    for i in range(10):
        xs, ys = sess.run([img_batch, label_batch])

        print(xs[2].shape)
    print(xs.shape)
    coord.request_stop()
    coord.join(threads)