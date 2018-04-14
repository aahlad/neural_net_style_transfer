"""
Package: EECS 692 Project
Title: vgg_19
Author: Aahlad Chandrabhatta
Description: vgg_19 class file
Date: March 2018
"""

import scipy.io as spio
import tensorflow as tf


class Vgg19:
    """ VGG 19 graph with pre-trained weights """
    def __init__(self, image_tensor=None):
        self.image_tensor = image_tensor
        self.create_layers()

    def create_layers(self):
        mat = spio.loadmat('imagenet-vgg-verydeep-19.mat')
        layers = mat['layers']

        self.conv1_1 = self.create_conv(prev_layer=self.image_tensor,
                                        name='conv1_1',
                                        weights_as_list=layers[0][0][0][0][2][0][0])
        self.relu1_1 = self.create_relu(prev_layer=self.conv1_1,
                                        name='relu1_1',
                                        biases_as_list=layers[0][0][0][0][2][0][1])
        self.conv1_2 = self.create_conv(prev_layer=self.relu1_1,
                                        name='conv1_2',
                                        weights_as_list=layers[0][2][0][0][2][0][0])
        self.relu1_2 = self.create_relu(prev_layer=self.conv1_2,
                                        name='relu1_2',
                                        biases_as_list=layers[0][2][0][0][2][0][1])
        self.pool1 = self.create_pool(prev_layer=self.relu1_2,
                                      name='pool1')
        self.conv2_1 = self.create_conv(prev_layer=self.pool1,
                                        name='conv2_1',
                                        weights_as_list=layers[0][5][0][0][2][0][0])
        self.relu2_1 = self.create_relu(prev_layer=self.conv2_1,
                                        name='relu2_1',
                                        biases_as_list=layers[0][5][0][0][2][0][1])
        self.conv2_2 = self.create_conv(prev_layer=self.relu2_1,
                                        name='conv2_2',
                                        weights_as_list=layers[0][7][0][0][2][0][0])
        self.relu2_2 = self.create_relu(prev_layer=self.conv2_2,
                                        name='relu2_2',
                                        biases_as_list=layers[0][7][0][0][2][0][1])
        self.pool2 = self.create_pool(prev_layer=self.relu2_2,
                                      name='pool2')
        self.conv3_1 = self.create_conv(prev_layer=self.pool2,
                                        name='conv3_1',
                                        weights_as_list=layers[0][10][0][0][2][0][0])
        self.relu3_1 = self.create_relu(prev_layer=self.conv3_1,
                                        name='relu3_1',
                                        biases_as_list=layers[0][10][0][0][2][0][1])
        self.conv3_2 = self.create_conv(prev_layer=self.relu3_1,
                                        name='conv3_2',
                                        weights_as_list=layers[0][12][0][0][2][0][0])
        self.relu3_2 = self.create_relu(prev_layer=self.conv3_2,
                                        name='relu3_2',
                                        biases_as_list=layers[0][12][0][0][2][0][1])
        self.conv3_3 = self.create_conv(prev_layer=self.relu3_2,
                                        name='conv3_3',
                                        weights_as_list=layers[0][14][0][0][2][0][0])
        self.relu3_3 = self.create_relu(prev_layer=self.conv3_3,
                                        name='relu3_3',
                                        biases_as_list=layers[0][14][0][0][2][0][1])
        self.conv3_4 = self.create_conv(prev_layer=self.relu3_3,
                                        name='conv3_4',
                                        weights_as_list=layers[0][16][0][0][2][0][0])
        self.relu3_4 = self.create_relu(prev_layer=self.conv3_4,
                                        name='relu3_4',
                                        biases_as_list=layers[0][16][0][0][2][0][1])
        self.pool3 = self.create_pool(prev_layer=self.relu3_4,
                                      name='pool3')
        self.conv4_1 = self.create_conv(prev_layer=self.pool3,
                                        name='conv4_1',
                                        weights_as_list=layers[0][19][0][0][2][0][0])
        self.relu4_1 = self.create_relu(prev_layer=self.conv4_1,
                                        name='relu4_1',
                                        biases_as_list=layers[0][19][0][0][2][0][1])
        self.conv4_2 = self.create_conv(prev_layer=self.relu4_1,
                                        name='conv4_2',
                                        weights_as_list=layers[0][21][0][0][2][0][0])
        self.relu4_2 = self.create_relu(prev_layer=self.conv4_2,
                                        name='relu4_2',
                                        biases_as_list=layers[0][21][0][0][2][0][1])
        self.conv4_3 = self.create_conv(prev_layer=self.relu4_2,
                                        name='conv4_3',
                                        weights_as_list=layers[0][23][0][0][2][0][0])
        self.relu4_3 = self.create_relu(prev_layer=self.conv4_3,
                                        name='relu4_3',
                                        biases_as_list=layers[0][23][0][0][2][0][1])
        self.conv4_4 = self.create_conv(prev_layer=self.relu4_3,
                                        name='conv4_4',
                                        weights_as_list=layers[0][25][0][0][2][0][0])
        self.relu4_4 = self.create_relu(prev_layer=self.conv4_4,
                                        name='relu4_4',
                                        biases_as_list=layers[0][25][0][0][2][0][1])
        self.pool4 = self.create_pool(prev_layer=self.relu4_4,
                                      name='pool4')
        self.conv5_1 = self.create_conv(prev_layer=self.pool4,
                                        name='conv5_1',
                                        weights_as_list=layers[0][28][0][0][2][0][0])
        self.relu5_1 = self.create_relu(prev_layer=self.conv5_1,
                                        name='relu5_1',
                                        biases_as_list=layers[0][28][0][0][2][0][1])
        self.conv5_2 = self.create_conv(prev_layer=self.relu5_1,
                                        name='conv5_2',
                                        weights_as_list=layers[0][30][0][0][2][0][0])
        self.relu5_2 = self.create_relu(prev_layer=self.conv5_2,
                                        name='relu5_2',
                                        biases_as_list=layers[0][30][0][0][2][0][1])
        self.conv5_3 = self.create_conv(prev_layer=self.relu5_2,
                                        name='conv5_3',
                                        weights_as_list=layers[0][32][0][0][2][0][0])
        self.relu5_3 = self.create_relu(prev_layer=self.conv5_3,
                                        name='relu5_3',
                                        biases_as_list=layers[0][32][0][0][2][0][1])
        self.conv5_4 = self.create_conv(prev_layer=self.relu5_3,
                                        name='conv5_4',
                                        weights_as_list=layers[0][34][0][0][2][0][0])
        self.relu5_4 = self.create_relu(prev_layer=self.conv5_4,
                                        name='relu5_4',
                                        biases_as_list=layers[0][34][0][0][2][0][1])
        self.pool5 = self.create_pool(prev_layer=self.relu5_4,
                                      name='pool5')

    def create_conv(self,
                    prev_layer=None,
                    name='',
                    weights_as_list=[]):
        with tf.variable_scope(name) as scope:
            weights = tf.constant(weights_as_list, name=name+'_weights')
            conv_layer = tf.nn.conv2d(input=prev_layer,
                                      filter=weights,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name=name)
            return conv_layer

    def create_relu(self,
                    prev_layer=None,
                    name='',
                    biases_as_list=[]):
        biases_as_list = biases_as_list.reshape(biases_as_list.size)
        with tf.variable_scope(name) as scope:
            biases = tf.constant(biases_as_list, name=name+'_biases')
            relu_layer = tf.nn.relu(prev_layer + biases)
            # relu_layer = prev_layer + biases
            return relu_layer

    def create_pool(self,
                    prev_layer=None,
                    name='',
                    is_avg=True):
        pooling_type = 'MAX'
        if is_avg:
            pooling_type = 'AVG'
        with tf.variable_scope(name) as scope:
            pool_layer = tf.nn.pool(input=prev_layer,  # TWEAK: nn.avg_pool?
                                    window_shape=[2, 2],
                                    pooling_type=pooling_type,
                                    padding='SAME',  # TWEAK: Same padding?
                                    strides=[2, 2],
                                    data_format='NHWC',
                                    name=name)
            return pool_layer
