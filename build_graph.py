#!/usr/bin/env python3 
import sys
try:
    import ujson as json
except:
    import json
from collections import Counter, defaultdict

try:
    from smart_open import smart_open as open
except:
    import open

import pickle


PATH_DATA = '/local2/jyunyu/rin/data/'

def get_id(node_id, x):
    if x not in node_id:
        node_id[x] = len(node_id) + 1
    return node_id[x]

def add_edge(node_id, ecnt, x, y):
    xid = get_id(node_id, x)
    yid = get_id(node_id, y)
    ecnt[(xid, yid)] += 1
    ecnt[(yid, xid)] += 1



if __name__ == '__main__':

    print('Calculating...', file=sys.stderr)
    nid = {}
    ecnt = Counter()
    with open(PATH_DATA + 'session.train', 'r') as fp:
        for line in fp: 
            data = line.strip().split('\t')
            queries = [data[i].strip() for i in range(1, len(data), 3)]
            sites = [data[i].strip() for i in range(2, len(data), 3)]
            assert(len(queries) == len(sites))
            
            for i in range(len(queries)):
                q = queries[i]
                # q-q
                if i > 0:
                    qp = queries[i - 1]
                    add_edge(nid, ecnt, ('query', q), ('query', qp))

                # q-t
                terms = q.split(' ')
                for t in terms:
                    add_edge(nid, ecnt, ('query', q), ('term', t))

                # t-t
                for j in range(1, len(terms)):
                    add_edge(nid, ecnt, ('term', terms[j-1]), ('term', terms[j]))

                # q-t
                site_list = sites[i].split('##') if sites[i] != '' else []
                for s in site_list:
                    add_edge(nid, ecnt, ('query', q), ('site', s))
    
    print('{} nodes and {} edges'.format(len(nid), len(ecnt)), file=sys.stderr)

    print('Dumping edge list...', file=sys.stderr)
    
    with open('model/edge.list', 'w') as wp:
        for edge in ecnt:
            print('{} {} {}'.format(edge[0], edge[1], ecnt[edge]), file=wp)

    print('Dumping objects...', file=sys.stderr)
    pickle.dump(nid, open('model/node_id.pkl', 'wb'))
    pickle.dump(ecnt, open('model/edge_cnt.pkl', 'wb'))

