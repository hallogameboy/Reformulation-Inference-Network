#!/usr/bin/env python3
import data_helpers
from rin import RIN
from scipy import spatial
from collections import defaultdict, Counter

def cosine_sim(__x, __y):
    return 1.0  - spatial.distance.cosine(__x, __y)

from scipy.stats import spearmanr

import pickle

import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import yaml, time, datetime
import os, sys, random
try:
    import ujson as json
except:
    print('Cannot import ujson, import json instead.', file=sys.stderr)
    import json
try:
    from smart_open import smart_open as open
except:
    print('smart_open inexists, use the original open instead.', file=sys.stderr)

def handle_flags():
    # data loading
    tf.flags.DEFINE_string('config', 'config.yml', 'configure file (default: config.yml)')
    # model parameters 
    tf.flags.DEFINE_integer('emb_dim', 256, 'Dimensionality of graph embedding')
    tf.flags.DEFINE_integer('max_len', 8, 'Max context lenth (default: 8)')
    tf.flags.DEFINE_float('valid_percent', 0.1, 'Percentage of training data for validation (default: 0.1)')
    tf.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of hidden layers (default 1)')
    tf.flags.DEFINE_integer('num_hidden_nodes', 256, 'Number of hidden layers (default 128)')
    tf.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability (default: 0.8)')
    tf.flags.DEFINE_float('l2_reg_lambda', 1e-3, 'L2 regularization lambda (default: 1e-3)')
    tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate while training (default: 1e-3)')
    tf.flags.DEFINE_integer('random_seed', 252, 'Random seeds for reproducibility (default: 252)')
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 1024)")
    tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 1000)")
    tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 10000")
    tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    # Set up flags
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
            print("{}={}".format(attr.upper(), value))
    return FLAGS


if __name__ == '__main__':

    #########################################
    #   Load arguments and configure file   #
    #########################################
    # Process CML arguments
    FLAGS = handle_flags()

    # Load the config file 
    cfg = yaml.load(open('config.yml', 'r'))

    #########################################
    #   Set up random seeds                 #
    #########################################
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    path_result = cfg['path_prediction']

    
    #########################################
    #  Load node ids and embedding          #
    #########################################

    print('Loading node ids and embedding', file=sys.stderr)
    node_id = pickle.load(open(cfg['path_nodeid'], 'rb'))
    node_emb = {}
    tq_emb = {}
    with open(cfg['path_emb'], 'r') as fp:
        fc = True
        for line in fp:
            if fc:
                _, n_emb = map(int, line.strip().split())
                assert(n_emb == FLAGS.emb_dim)
                fc = False
            else:
                data = line.strip().split()
                assert(len(data) == n_emb + 1)
                node_emb[int(data[0])] = np.array([float(x) for x in data[1:]])
    

    qc_loc = defaultdict(int)
    qc = defaultdict(list)
    with open(cfg['path_data'] + 'candidates.suggestion', 'r') as fp:
        for line in fp:
            data = json.loads(line)
            q = data['query']
            loc = 0
            for c in data['candidates']:
                loc += 1
                qc_loc[(q, c)] = loc
                qc[q].append(c)
                if loc >= 20:
                    break
            # assert(len(qc[q]) == 10)

    
    #########################################
    #  Load training and testing data       #
    #########################################
    
    print('Loading training and testing data', file=sys.stderr)

    #train_ses, valid_ses = data_helpers.load_train_valid_light(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, cfg['path_data'] + 'session.train')
    train_ses, valid_ses = data_helpers.load_train_valid_light(FLAGS, cfg['path_data'] + 'session.train')
    print('{} training and {} validation sessions.'.format(len(train_ses), len(valid_ses)), file=sys.stderr)


    test_ses = data_helpers.load_data_light(FLAGS, cfg['path_data'] + 'session.test')
    print('{} testing sessions.'.format(len(test_ses)), file=sys.stderr)


    
    #train_y *= 10.0
    #test_y *= 10.0
    #valid_y *= 10.0


    #########################################
    #  Create and train the model           #
    #########################################
    
    with tf.Graph().as_default():
        # set up session configuration
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        # get into training session
        with sess.as_default():
            # construct a model with parameters 
            model = RIN(
                    seq_len=FLAGS.max_len,
                    emb_size=FLAGS.emb_dim,
                    num_hidden_layers=FLAGS.num_hidden_layers,
                    num_hidden_nodes=FLAGS.num_hidden_nodes,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            # set up training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.pardir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            # if not os.path.exists(checkpoint_dir):
            #     os.makedirs(checkpoint_dir)
            # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            print('Global var initializer')
            sess.run(tf.global_variables_initializer())
            
            # Define the batch train step        
            def train_step(X_batch, R_batch, y_batch, rl_batch):
                feed_dict = {
                        model.X: X_batch,
                        model.R: R_batch,
                        model.y: y_batch,
                        model.rl: rl_batch,
                        model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }   
                _, step, loss = sess.run(
                        [train_op, global_step, model.loss],
                        feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 10 == 0:
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
            
            
            best_error = 1e+100

            def dev_step(X_batch, R_batch, y_batch, rl_batch):
                N_BATCH = 8192
                total_loss = 0.0
                total_cnt = 0.0
                for i in range(0, len(X_batch), N_BATCH):
                    feed_dict = {
                            model.X: X_batch[i:(i+N_BATCH)],
                            model.R: R_batch[i:(i+N_BATCH)],
                            model.y: y_batch[i:(i+N_BATCH)],
                            model.rl: rl_batch[i:(i+N_BATCH)],
                        model.dropout_keep_prob: 1.0
                    }   
                    step, loss = sess.run([global_step, model.loss], feed_dict)
                    N = float(min(len(X_batch) - i, N_BATCH))
                    total_loss += N * loss
                    total_cnt  += N

                loss = total_loss / total_cnt
                time_str = datetime.datetime.now().isoformat()
                # print("Validation {}: step {}, loss {:g}".format(time_str, step, loss))
                return loss

            # Batch training
            for n_epochs in range(FLAGS.num_epochs):
                random.shuffle(train_ses)
                for batch_L in range(0, len(train_ses), FLAGS.batch_size):
                    X_batch, R_batch, y_batch, rl_batch = data_helpers.load_sessions(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, train_ses[batch_L:(batch_L+FLAGS.batch_size)])
                    train_step(X_batch, R_batch, y_batch, rl_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.checkpoint_every == 0:
                        valid_loss = []
                        for i in range(0, len(valid_ses), FLAGS.batch_size * 400):
                            valid_X, valid_R, valid_y, valid_rl = data_helpers.load_sessions(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, valid_ses[i:(i+FLAGS.batch_size * 400)])
                            valid_loss.append(dev_step(valid_X, valid_R, valid_y, valid_rl))
                        valid_loss = np.mean(valid_loss)
                        valid_X, valid_R, valid_y, valid_rl = [], [], [], []
                        time_str = datetime.datetime.now().isoformat()
                        print("Validation {}: step {}, loss {:g}".format(time_str, current_step, valid_loss))

                        if valid_loss < best_error:
                            best_error = valid_loss
                            print('best_error -> {}'.format(best_error), file=sys.stderr)

                            test_losses = []
                            test_pred = None
                            N_BATCH = 8192
                            for j in range(0, len(test_ses), FLAGS.batch_size * 400):
                                test_X, test_R, test_y, test_rl = data_helpers.load_sessions(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, test_ses[j:(j+FLAGS.batch_size * 400)])
                                for i in range(0, len(test_X), N_BATCH):
                                    feed_dict = {
                                            model.X: test_X[i:(i+N_BATCH)],
                                            model.R: test_R[i:(i+N_BATCH)],
                                            model.y: test_y[i:(i+N_BATCH)],
                                            model.rl: test_rl[i:(i+N_BATCH)],
                                            model.dropout_keep_prob: 1.0
                                        }

                                    step, pred, loss = sess.run([global_step, model.y_hat, model.loss], feed_dict)
                                    test_losses.append(loss)
                                    test_pred = pred if i + j == 0 else np.append(test_pred, pred, axis=0)
                            print('test_error -> {}'.format(np.mean(test_losses)), file=sys.stderr)
                            np.save(path_result, test_pred)

            '''
