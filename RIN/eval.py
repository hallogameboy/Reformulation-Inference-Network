#!/usr/bin/env python3
import data_helpers
from rin import RIN
from scipy import spatial
from collections import defaultdict, Counter
import math

from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean

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

import multiprocessing
from multiprocessing import Pool


def handle_flags():
    # data loading
    tf.flags.DEFINE_string('config', 'config.yml', 'configure file (default: config.yml)')
    # model parameters 
    tf.flags.DEFINE_integer('emb_dim', 256, 'Dimensionality of graph embedding')
    tf.flags.DEFINE_integer('max_len', 8, 'Max context lenth (default: 8)')
    tf.flags.DEFINE_float('valid_percent', 0.1, 'Percentage of training data for validation (default: 0.1)')
    tf.flags.DEFINE_integer('num_hidden_layers', 1, 'Number of hidden layers (default 1)')
    tf.flags.DEFINE_integer('num_hidden_nodes', 128, 'Number of hidden layers (default 128)')
    tf.flags.DEFINE_float('dropout_keep_prob', 0.8, 'Dropout keep probability (default: 0.8)')
    tf.flags.DEFINE_float('l2_reg_lambda', 1e-3, 'L2 regularization lambda (default: 1e-3)')
    tf.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate while training (default: 1e-3)')
    tf.flags.DEFINE_integer('random_seed', 252, 'Random seeds for reproducibility (default: 252)')
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 1024)")
    tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 1000)")
    tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100")
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

# Load the config file 
cfg = yaml.load(open('config.yml', 'r'))

node_id = pickle.load(open(cfg['path_nodeid'], 'rb'))
node_emb = {}
tq_emb = {}


def pp_get_tq_emb(q):
    if q not in tq_emb:
        ts = q.split(' ')
        emb = np.zeros(FLAGS.emb_dim)
        for t in ts:
            if ('term', t) in node_id:
                emb += node_emb[node_id[('term', t)]]
        tq_emb[q] = emb
        #np.append(emb, [math.log10(len(ts) + 1)])
    return tq_emb[q]

#########################################
#   Load arguments and configure file   #
#########################################
# Process CML arguments
FLAGS = handle_flags()

if __name__ == '__main__':


    num_cores = multiprocessing.cpu_count()



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


    print('Loading predictions', file=sys.stderr)
    pred = np.load(path_result + '.npy')
    print(pred)
    
    mrr_list = [[], [], [], []]
    sr1_list = [[], [], [], []]
    print('Calculating', file=sys.stderr)
    with open(cfg['path_data'] + 'session.test', 'r') as fp:
        lc = 0
        plc = 0
        for line in fp:
            lc += 1
            if lc % 10 == 0:
                sys.stderr.write('\r{} lines'.format(lc))

            data = line.strip().split('\t')
            qs = [data[i] for i in range(1, len(data), 3)]
            for i in range(1, len(qs)):
                if qc_loc[(qs[i-1], qs[i])] == 0:
                    continue
                L = len(qc[qs[i-1]])
                scores = [float(x) for x in pred[plc:(plc + L)]]
                plc += L

                can_list = sorted(list(zip(qc[qs[i-1]], scores)), key=lambda x: -x[-1])
                loc = 0
                for x in can_list:
                    loc += 1
                    if x[0] == qs[i]:
                        break

                mrr = 1.0 / float(loc) if loc > 0 else 0.0
                sr1 = 1.0 if loc == 1 else 0.0
                
                mrr_list[0].append(mrr)
                sr1_list[0].append(sr1)

                if i == 1:
                    mrr_list[1].append(mrr)
                    sr1_list[1].append(sr1)
                elif i < 4:
                    mrr_list[2].append(mrr)
                    sr1_list[2].append(sr1)
                else:
                    mrr_list[3].append(mrr)
                    sr1_list[3].append(sr1)

    for i in range(4):
        print(np.mean(mrr_list[i]))
        print(np.mean(sr1_list[i]))
