# -*- coding: utf-8 -*-

# Sample code to use string producer.

import numpy as np
import tensorflow as tf

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

n_input = 128 * 128

with tf.variable_scope('ConvNet', reuse=False):
    X = tf.placeholder(tf.float32,[None, 128, 128, 1])
    o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
    o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
    o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
    o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)
    o5 = tf.layers.conv2d(inputs=o4, filters=128, kernel_size=3, activation=tf.nn.relu)
    o6 = tf.layers.max_pooling2d(inputs=o5, pool_size=2, strides=2)
    o7 = tf.layers.conv2d(inputs=o6, filters=256, kernel_size=3, activation=tf.nn.relu)
    o8 = tf.layers.max_pooling2d(inputs=o7, pool_size=2, strides=2)
    o9 = tf.layers.conv2d(inputs=o8, filters=512, kernel_size=3, activation=tf.nn.relu)
    o10 = tf.layers.max_pooling2d(inputs=o9, pool_size=2, strides=2)
    # tf.layers.flatten(o10)
    # units 30 o 5
    h = tf.layers.dense(inputs=tf.reshape(o10, [1, 2 * 2 * 512]), units=100, activation=tf.nn.relu)
    y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

# --------------------------------------------------
#
#       PLAY
#
# --------------------------------------------------

import cv2

cap = cv2.VideoCapture(0)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "./tmp/model.ckpt")
    print("Model restored.")

    while True:
        ret, img = cap.read()  # 720x1280x3 <-- print(img.shape);

        resized = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        cropped = resized[0:180, 70:250]
        resized64 = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_AREA)
        gray = np.asarray(cv2.cvtColor(resized, 7))

        cv2.imshow('Capture', gray)
        frame = gray.reshape(-1, 128, 128, 1)
        print (sess.run(y, feed_dict={X: frame}))

        ch = 0xFF & cv2.waitKey(1)
        if ch == 27:
            break

cv2.destroyAllWindows()
