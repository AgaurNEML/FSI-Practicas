# -*- coding: utf-8 -*-

# Sample code to use string producer.

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h


num_classes = 3
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        # Ajustar las imagenes
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(i, num_classes)
        image = tf.reshape(image, [128, 128, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
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

        #units 30 o 5
        h = tf.layers.dense(inputs=tf.reshape(o10, [batch_size * 3, 2 * 2 * 512]), units=100, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["muestras/train/0/*.jpg", "muestras/train/1/*.jpg", "muestras/train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["muestras/valid/0/*.jpg", "muestras/valid/1/*.jpg", "muestras/valid/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["muestras/test/0/*.jpg", "muestras/test/1/*.jpg", "muestras/test/2/*.jpg"], batch_size=batch_size)



example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, tf.float32)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)



# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    error_valid = 0
    errors_valid = []
    previousError = 0

    for epoch in range(400):
        sess.run(optimizer)
        error_valid = sess.run(cost_valid)
        errors_valid.append(error_valid)

        if epoch % 20 == 0:
            print("\nIter:", epoch, "\n---------------------------------------------")

            print(sess.run(label_batch_train))
            print(sess.run(example_batch_train_predicted))
            print("Error de validacion:", error_valid)
            print("Error del entreamiento:", sess.run(cost))

         #Controlamos el error

        #if np.absolute(error_valid - previousError) < 0.001:
        #    epocaFinal = epoch
        #    break
        #    print("Final Epoch: ", epocaFinal)
        #else:
        #    previousError = errors_valid[len(errors_valid) - 2]

    print("Final Epoch: ", epoch)

    print("\n- - - - - - - - -")
    print("\n   Toca testear  ")
    print("\n- - - - - - - - -")

    conjunto_total = 0.0
    fallo = 0.0

    conjunto_test = sess.run(label_batch_test)

    conjunto_test_esperado = sess.run(example_batch_test_predicted)

    for muestra, muestra_ideal in zip(conjunto_test, conjunto_test_esperado):
        if np.argmax(conjunto_test) != np.argmax(conjunto_test_esperado):
            fallo += 1
        conjunto_total += 1

    print("\nLa tasa de fallos es; ")
    print('{porcentaje:.2%}'.format(porcentaje=fallo/conjunto_total))

    print("\nLa tasa de exito es; ")
    print('{porcentaje:.2%}'.format(porcentaje=1-(fallo/conjunto_total)))



    plt.title("Error de validacion")
    plt.plot(errors_valid)
    plt.show()

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)