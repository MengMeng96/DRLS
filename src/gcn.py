"""
Graph Convolutional Network

Propergate node features among neighbors
via parameterized message passing scheme
"""

import copy
import numpy as np
import tensorflow as tf
from tf_op import glorot, ones, zeros

class GraphCNN(object):
    def __init__(self, inputs, input_dim, hid_dims, output_dim,
                 max_depth, act_fn, scope='gcn'):

        self.inputs = inputs

        self.input_dim = input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim

        self.max_depth = max_depth

        self.act_fn = act_fn
        self.scope = scope

        # valid edges
        # self.valid_edges = tf.placeholder(tf.int32, [None])

        # message passing
        # self.masks = tf.placeholder(tf.float32, [None, None, 1])
        self.reachable_edge_matrix = tf.placeholder(tf.float32, [None, None, None])

        # initialize message passing transformation parameters
        # h: x -> x'
        # hid_dims = [16,8], output_dim = 8
        print("input_dim", self.input_dim)
        self.prep_weights, self.prep_bias = \
            self.init(self.input_dim, self.hid_dims, self.output_dim)

        # f: x' -> e
        # self.proc_weights, self.proc_bias = \
        #     self.init(self.output_dim, self.hid_dims, self.output_dim)

        # g: e -> e
        self.agg_weights, self.agg_bias = \
            self.init(self.output_dim, self.hid_dims, self.output_dim)

        # graph message passing
        self.outputs = self.forward()

    def init(self, input_dim, hid_dims, output_dim):
        # Initialize the parameters
        # these weights may need to be re-used
        # e.g., we may want to propagate information multiple times
        # but using the same way of processing the nodes
        weights = []
        bias = []

        curr_in_dim = input_dim

        # hidden layers
        for hid_dim in hid_dims:
            weights.append(
                glorot([curr_in_dim, hid_dim], scope=self.scope))
            bias.append(
                zeros([hid_dim], scope=self.scope))
            curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, output_dim], scope=self.scope))
        bias.append(zeros([output_dim], scope=self.scope))

        return weights, bias

    def forward(self):
        # message passing among nodes
        # the information is flowing from leaves to roots
        x = self.inputs
        # raise x into higher dimension
        for l in range(len(self.prep_weights)):
            x = tf.matmul(x, self.prep_weights[l])
            x += self.prep_bias[l]
            x = self.act_fn(x)

        for d in range(self.max_depth):
            # work flow: index_select -> f -> masked assemble via adj_mat -> g
            y = x

            # aggregate child features
            for l in range(len(self.agg_weights)):
                y = tf.matmul(y, self.agg_weights[l])
                y += self.agg_bias[l]
                y = self.act_fn(y)

            # remove the artifact from the bias term in g
            # n * n, n * 3 --- n * 3
            y = tf.matmul(self.reachable_edge_matrix[d], y)

            # assemble neighboring information
            x = x + y

        return x
