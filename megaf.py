"""
Package: EECS 692 Project
Title: oooola
Author: Aahlad Chandrabhatta
Description: ooolala
Date: March 2018
Issues: 1) Mean Centering
        2) White noise mix with content
        3) Biases addition
        4) Positive gradients
        5) White noise image initialization
        6) Style and Content weights
TODO: 1) Loss functions
"""

import sys
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from vgg_19 import Vgg19

SIZE = (225, 300)  # Height followed by width
VGG_MEAN = np.asarray([123.68, 116.779, 103.939]).astype(np.float32)

"""
import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
x = tf.add(a, b, name='add')

writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x))

writer.close()
"""


def image_to_tensor(image_file='', name=''):
    image = Image.open(image_file)
    image2 = ImageOps.fit(image, SIZE[::-1], Image.ANTIALIAS)
    image = np.asarray(image2, np.float32) - VGG_MEAN
    # import pdb; pdb.set_trace()
    # boss = Image.fromarray(image, 'RGB')
    # boss.save('boss_'+name+'un.jpg')
    # boss = Image.fromarray(np.array(image, np.uint8), 'RGB')
    # boss.save('boss_'+name+'un_np.jpg')
    # boss2 = Image.fromarray(image + VGG_MEAN, 'RGB')
    # boss2.save('boss_'+name+'nu.jpg')
    # boss2 = Image.fromarray(np.array(image + VGG_MEAN, np.uint8), 'RGB')
    # boss2.save('boss_'+name+'nu_np.jpg')
    image = np.array([image])
    with tf.variable_scope(name) as scope:
        image_tensor = tf.constant(image, name=name)
        return image_tensor


def white_noise_image(pseudo=None):
    # image = np.random.uniform(256, size=(1, SIZE[0], SIZE[1], 3)).astype(np.float32)
    image = np.random.uniform(-20, 20, (1, SIZE[0], SIZE[1], 3)).astype(np.float32)
    # img = Image.fromarray(image[0], 'RGB')
    # img.save('wn.jpg')
    # img.show()
    white_noise_initializer = tf.constant(image, name='white_noise_initializer')
    with tf.variable_scope('white_noise_image') as scope:
        image_tensor = None
        if pseudo is None:
            image_tensor = tf.get_variable(name='white_noise_image',
                                           initializer=white_noise_initializer)
        else:
            image_tensor = tf.get_variable(name='white_noise_image',
                                           initializer=pseudo)
        return image_tensor


def overall_loss(vgg_train, vgg_content, vgg_style):
    with tf.variable_scope('content_loss') as scope:
        content_loss = tf.reduce_sum(((vgg_train.conv4_2 - vgg_content.conv4_2)**2)/2)

    with tf.variable_scope('style_loss') as scope:
        gm_train = []
        gm_style = []
        train_layers = [vgg_train.conv1_1, vgg_train.conv2_1, vgg_train.conv3_1, vgg_train.conv4_1, vgg_train.conv5_1]
        # train_layers = [vgg_train.conv1_1]
        style_layers = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1, vgg_style.conv5_1]
        # style_layers = [vgg_style.conv1_1]
        style_layer_loss = []
        style_loss = 0
        for layer in train_layers:
            F = tf.reshape(layer, (layer.shape[1]*layer.shape[2], layer.shape[3]))
            gm_train.append(tf.matmul(tf.transpose(F), F))

        for layer in style_layers:
            F = tf.reshape(layer, (layer.shape[1]*layer.shape[2], layer.shape[3]))
            gm_style.append(tf.matmul(tf.transpose(F), F))

        for i in range(len(gm_train)):
            G = gm_train[i]
            A = gm_style[i]
            N = int(train_layers[i].shape[3])
            M = int(train_layers[i].shape[1])*int(train_layers[i].shape[2])
            style_layer_loss.append(tf.reduce_sum(((G-A)**2)/((2*N*M)**2)))

        for i in range(len(style_layer_loss)):
            style_loss += style_layer_loss[i]/len(style_layer_loss)

    with tf.variable_scope('total_loss') as scope:
        return content_loss*(1) + style_loss*(1000)


if __name__ == '__main__':
    content_image_tensor = image_to_tensor(image_file=sys.argv[1], name='content')
    style_image_tensor = image_to_tensor(image_file=sys.argv[2], name='style')
    # white_noise_tensor = white_noise_image(pseudo=content_image_tensor)
    white_noise_tensor = white_noise_image()
    vgg_content = Vgg19(image_tensor=content_image_tensor)
    vgg_train = Vgg19(image_tensor=white_noise_tensor)
    vgg_style = Vgg19(image_tensor=style_image_tensor)
    loss = overall_loss(vgg_train, vgg_content, vgg_style)
    opt = tf.train.AdamOptimizer(2.0).minimize(loss)
    # opt = tf.train.AdadeltaOptimizer(learning_rate=1, rho=0.95, epsilon=1e-6, use_locking=False, name='AdaDelta').minimize(loss)

    writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # wni_boss = sess.run([white_noise_tensor])[0]
        # wni_boss_1 = Image.fromarray(np.asarray(wni_boss[0], np.int), 'RGB')
        # wni_boss_1.save('wn_boss_'+'un.jpg')
        # wni_boss_2 = Image.fromarray(np.asarray(wni_boss[0] + VGG_MEAN, np.int), 'RGB')
        # wni_boss_2.save('wn_boss_'+'nu.jpg')
        # wni_boss = sess.run([content_image_tensor])[0]
        # wni_boss_1 = Image.fromarray(np.asarray(wni_boss[0], np.int), 'RGB')
        # wni_boss_1.save('1wn_boss_'+'un.jpg')
        # wni_boss_2 = Image.fromarray(np.asarray(wni_boss[0] + VGG_MEAN, np.int), 'RGB')
        # wni_boss_2.save('1wn_boss_'+'nu.jpg')
        # wni_boss = sess.run([style_image_tensor])[0]
        # wni_boss_1 = Image.fromarray(np.asarray(wni_boss[0], np.int), 'RGB')
        # wni_boss_1.save('2wn_boss_'+'un.jpg')
        # wni_boss_2 = Image.fromarray(np.asarray(wni_boss[0] + VGG_MEAN, np.int), 'RGB')
        # wni_boss_2.save('2wn_boss_'+'nu.jpg')
        for _ in range(100000):
            __, wni, _loss = sess.run([opt, white_noise_tensor, loss])
            print(_, _loss)
            if _%10 == 0:
                # wni_1 = Image.fromarray(np.asarray(wni[0], np.uint8), 'RGB')
                # wni_1.save('wn_'+str(_)+'un.jpg')
                wni_2 = Image.fromarray(np.asarray(wni[0] + VGG_MEAN, np.uint8), 'RGB')
                wni_2.save('wn_'+str(_)+'.jpg')

        # pool5_out1 = sess.run(vgg_content.pool5)
        # pool5_out2 = sess.run(vgg_style.pool5)
        # pool5_out3 = sess.run(vgg_train.pool5)
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
        # for _ in range(5):
            # print(sess.run(x))
            # print(sess.run(tf.multiply(a1, b1)))

    writer.close()
    # print(pool5_out1)
    # print(pool5_out2)
    # print(pool5_out3)
