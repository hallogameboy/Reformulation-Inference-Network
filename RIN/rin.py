#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers

class RIN(object):
    def __init__(self
            , seq_len
            , emb_size
            , num_hidden_layers = 1
            , num_hidden_nodes = 128
            , l2_reg_lambda = 0.0):
        
        # Set parameters
        self.seq_len = seq_len
        self.emb_size = emb_size
        # self.emb_cat_num = 1 * self.emb_size + 1
        self.emb_cat_num = 2 * self.emb_size + 2
        self.fea_len =  seq_len * self.emb_cat_num
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_nodes = num_hidden_nodes
        self.l2_reg_lambda = l2_reg_lambda
        
        # Build model
        self.build_model()


    def build_model(self):
        # placeholders for input and dropout
        self.X = tf.placeholder(tf.float32, [None, self.fea_len], name='X')
        self.R = tf.placeholder(tf.float32, [None, self.emb_cat_num], name='R')
        self.y = tf.placeholder(tf.float32, [None, 1], name='y')
        self.rl = tf.placeholder(tf.float32, [None, self.emb_size + 1], name='rl')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")        
        self.l2_loss = tf.constant(0.0)
        # RNN
        self.build_RNN()
        # hidden layers
        self.build_hidden_layers()            
        self.build_reformulation_influencer()
        self.pred_loss = -tf.reduce_mean(10.0 * tf.multiply(self.y, tf.log(self.y_hat + 1e-9)) + tf.multiply(1.0 - self.y, tf.log(1.0 - self.y_hat + 1e-9)) * 1.0 )
        self.ref_loss = tf.reduce_mean(tf.norm(self.rl - self.rl_hat))
        self.loss = self.pred_loss + self.ref_loss * 0.05 + self.l2_reg_lambda * self.l2_loss
        self.loss = tf.add_n([self.loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    
    def build_RNN(self):
        self.reshape_X = tf.reshape(self.X, [-1, self.seq_len, self.emb_cat_num])

        with tf.name_scope("rnn") :
            fw_cell = tf.contrib.rnn.GRUCell(self.num_hidden_nodes)
            fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_cell = tf.contrib.rnn.GRUCell(self.num_hidden_nodes)
            bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=self.reshape_X, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)

            attention_context = tf.get_variable(shape=[self.num_hidden_nodes*2], dtype=tf.float32, name='attention_context')
            projection = layers.fully_connected(outputs, self.num_hidden_nodes*2, activation_fn=tf.tanh, weights_regularizer=layers.l2_regularizer(scale=self.l2_reg_lambda))
            self.l2_loss += tf.nn.l2_loss(attention_context)
            attention_vector = tf.reduce_sum(tf.multiply(projection, attention_context), axis=2, keep_dims=True)
            weights = tf.nn.softmax(attention_vector, dim=1)
            weighted_projection = tf.multiply(outputs, weights)
            self.lstm_output = tf.reduce_sum(weighted_projection, axis=1)
            

    def build_reformulation_influencer(self):
        self.final_features = tf.concat([self.lstm_output, self.R], axis=1)
        print(self.final_features.get_shape())

        last_output = self.final_features
        self.hidden_layer_output = [last_output]
        
        last_size = int(self.final_features.get_shape()[-1])
        self.hidden_layer_output = []
        for i in range(self.num_hidden_layers):
            with tf.name_scope('ri-layer-%d' % i):
                W = tf.Variable(tf.truncated_normal([last_size, self.num_hidden_nodes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_hidden_nodes]), name='b')
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                output = tf.matmul(self.hidden_layer_output[-1] if i > 0 else self.final_features, W) + b
                output = tf.nn.elu(output)
                output = tf.nn.dropout(output, self.dropout_keep_prob)
                self.hidden_layer_output.append(output)
                last_size = self.num_hidden_nodes
                last_output = output

        with tf.name_scope('ri-final'):
            W = tf.Variable(tf.truncated_normal([int(last_output.get_shape()[-1]), self.emb_size + 1], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[self.emb_size + 1]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.rl_hat = tf.add(tf.matmul(self.hidden_layer_output[-1], W),  b, name='pred')


    def build_hidden_layers(self):
        self.final_features = tf.concat([self.lstm_output, self.R], axis=1)
        print(self.final_features.get_shape())

        last_output = self.final_features
        self.hidden_layer_output = [last_output]
        
        last_size = int(self.final_features.get_shape()[-1])
        self.hidden_layer_output = []
        for i in range(self.num_hidden_layers):
            with tf.name_scope('layer-%d' % i):
                W = tf.Variable(tf.truncated_normal([last_size, self.num_hidden_nodes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_hidden_nodes]), name='b')
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
                output = tf.matmul(self.hidden_layer_output[-1] if i > 0 else self.final_features, W) + b
                output = tf.nn.elu(output)
                output = tf.nn.dropout(output, self.dropout_keep_prob)
                self.hidden_layer_output.append(output)
                last_size = self.num_hidden_nodes
                last_output = output

        with tf.name_scope('final'):
            W = tf.Variable(tf.truncated_normal([int(last_output.get_shape()[-1]), 1], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[1]), name='b')
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.y_hat = tf.sigmoid(tf.add(tf.matmul(self.hidden_layer_output[-1], W),  b), name='pred')



