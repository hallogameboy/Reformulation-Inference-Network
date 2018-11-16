#!/usr/bin/env python3
import sys, os
import numpy as np
import pickle
import math
try:
    import ujson as json
except:
    print('Cannot import ujson, import json instead.', file=sys.stderr)
    import json

try:
    from smart_open import smart_open as open
except:
    print('smart_open inexists, use original one instead.', file=sys.stderr)
    
import tensorflow as tf
import random
  
def get_tq_emb(node_id, node_emb, tq_emb, FLAGS, q):
    if q not in tq_emb:
        ts = q.split(' ')
        emb = np.zeros(FLAGS.emb_dim)
        for t in ts:
            if ('term', t) in node_id:
                emb += node_emb[node_id[('term', t)]]
        tq_emb[q] = np.append(emb, [math.log10(len(ts) + 1)])
    return tq_emb[q]


def load_train_valid_light(FLAGS, filename):
    train, valid = [], []
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            queries = [data[i] for i in range(1, len(data), 3)]
            is_valid = True if random.random() < FLAGS.valid_percent else False
            if is_valid:
                valid.append(queries)
            else:
                train.append(queries)
    return train, valid


def load_data_light(FLAGS, filename):
    output = []
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            queries = [data[i] for i in range(1, len(data), 3)]
            output.append(queries)
    return output


def load_sessions(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, sessions):
    output_X, output_R, output_y, output_rl = [], [], [], []
    # n_fea = 1 * FLAGS.emb_dim + 1
    n_fea = 2 * FLAGS.emb_dim + 2
    for queries in sessions:
        embs = [get_tq_emb(node_id, node_emb, tq_emb, FLAGS, q) for q in queries]
        # qembs = [ node_emb[node_id[('query', q)]] if ('query', q) in node_id else np.zeros(FLAGS.emb_dim) for q in queries]
        data = np.array([])
        for i in range(len(queries)):
            data = np.append(data, embs[i])
            data = np.append(data, embs[i] - embs[i-1] if i > 0 else np.zeros(FLAGS.emb_dim))                
        for i in range(1, len(queries)):
            if (queries[i-1], queries[i]) not in qc_loc:
                continue

            L = max(0, (i - FLAGS.max_len)) * n_fea
            R = i * n_fea
            X = data[L:R]
            if X.size < n_fea * FLAGS.max_len:
                X = np.append(np.zeros(n_fea*FLAGS.max_len - X.size), X)
            assert(X.size == n_fea * FLAGS.max_len)
            rl = embs[i] - embs[i-1]

            for c in qc[queries[i-1]]:
                R = np.array([])
                cemb = get_tq_emb(node_id, node_emb, tq_emb, FLAGS, c)
                R = np.append(R, cemb)
                R = np.append(R, cemb - embs[i-1])

                output_X.append(X)
                output_R.append(R)
                output_y.append(1.0 if c == queries[i] else 0.0)
                output_rl.append(rl)

    output_y = np.array(output_y).reshape((len(output_y), 1))

    return np.array(output_X), np.array(output_R), output_y, np.array(output_rl)



def load_train_valid(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, filename):
    train_X, train_R, train_y = [], [], []
    valid_X, valid_R, valid_y = [], [], []
    n_fea = 2 * FLAGS.emb_dim  + 2
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            queries = [data[i] for i in range(1, len(data), 3)]
            embs = [get_tq_emb(node_id, node_emb, tq_emb, FLAGS, q) for q in queries]
            # qembs = [ node_emb[node_id[('query', q)]] if ('query', q) in node_id else np.zeros(FLAGS.emb_dim) for q in queries]
            data = np.array([])
            for i in range(len(queries)):
                # data = np.append(data, [math.log10(i + 1)])
                # data = np.append(data, qembs[i])
                data = np.append(data, embs[i])
                data = np.append(data, (embs[i] - embs[i-1]) if i > 0 else np.zeros(FLAGS.emb_dim))
            for i in range(1, len(queries)):
                if (queries[i-1], queries[i]) not in qc_loc:
                    continue
                is_valid = True if random.random() < FLAGS.valid_percent else False

                L = max(0, (i - FLAGS.max_len)) * n_fea
                R = i * n_fea
                X = data[L:R]
                if X.size < n_fea * FLAGS.max_len:
                    X = np.append(np.zeros(n_fea*FLAGS.max_len - X.size), X)
                assert(X.size == n_fea * FLAGS.max_len)

                for c in qc[queries[i - 1]]:
                    R = np.array([])
                    cemb = get_tq_emb(node_id, node_emb, tq_emb, FLAGS, c)
                    R = np.append(R, cemb)
                    R = np.append(R, cemb - embs[i-1])

                    if is_valid:
                        valid_X.append(X)
                        valid_R.append(R)
                        valid_y.append(1.0 if c == queries[i] else 0.0)
                    else:
                        train_X.append(X)
                        train_R.append(R)
                        train_y.append(1.0 if c == queries[i] else 0.0)
    
    valid_y = np.array(valid_y).reshape((len(valid_y), 1))
    train_y = np.array(train_y).reshape((len(train_y), 1))

    valid_X = np.array(valid_X)
    train_X = np.array(train_X)
    valid_R = np.array(valid_R)
    train_R = np.array(train_R)
    
    return train_X, train_R, train_y, valid_X, valid_R, valid_y

def load_data(node_id, node_emb, tq_emb, qc, qc_loc, FLAGS, filename):
    output_X, output_R, output_y = [], [], []
    n_fea = 2 * FLAGS.emb_dim + 2
    with open(filename, 'r') as fp:
        for line in fp:
            data = line.strip().split('\t')
            queries = [data[i] for i in range(1, len(data), 3)]
            embs = [get_tq_emb(node_id, node_emb, tq_emb, FLAGS, q) for q in queries]
            # qembs = [ node_emb[node_id[('query', q)]] if ('query', q) in node_id else np.zeros(FLAGS.emb_dim) for q in queries]
            data = np.array([])
            for i in range(len(queries)):
                # data = np.append(data, [math.log10(i + 1)])
                # data = np.append(data, qembs[i])
                data = np.append(data, embs[i])
                data = np.append(data, embs[i] - embs[i-1] if i > 0 else np.zeros(FLAGS.emb_dim))                
            for i in range(1, len(queries)):
                if (queries[i-1], queries[i]) not in qc_loc:
                    continue

                L = max(0, (i - FLAGS.max_len)) * n_fea
                R = i * n_fea
                X = data[L:R]
                if X.size < n_fea * FLAGS.max_len:
                    X = np.append(np.zeros(n_fea*FLAGS.max_len - X.size), X)
                assert(X.size == n_fea * FLAGS.max_len)

                for c in qc[queries[i-1]]:
                    R = np.array([])
                    cemb = get_tq_emb(node_id, node_emb, tq_emb, FLAGS, c)
                    R = np.append(R, cemb)
                    R = np.append(R, cemb - embs[i-1])

                    output_X.append(X)
                    output_R.append(R)
                    output_y.append(1.0 if c == queries[i] else 0.0)

    output_y = np.array(output_y).reshape((len(output_y), 1))

    return np.array(output_X), np.array(output_R), output_y



def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-1.0, 1.0, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    embedding_vectors[0] = 0.0
    return embedding_vectors


def batch_iter(data_, batch_size, num_epochs, shuffle=True):
    data = np.array(data_)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
