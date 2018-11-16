# Reformulation-Inference-Network

## Required Packages

* Python3 (or a compatible version)
* Tensorflow 1.4 (or a compatible version)
* NumPy
* SciPy
* Yaml
* ujson
* smart_open

## Quick Run

TODO

## General Instructions

1. Construct the heterogenous network of queries, terms, and websites based on the search log.
2. Train the embedding for each node in the network, e.g., using node2vec.
3. Train the Reformulation Inference Network.
4. Evaluate and dump the predicted scores.

## Query Session Preprocessing

The query logs should be prepreocessed into two files, including data/session.train and data/session.test, with the following plain text format. Note that all of cleaning processes should be accomplished before generaing the session data.

```
...
ID<tab>query1<tab>timestamp1<tab>website1-1#website1-2#website1-3...<tab>query2<tab>timestamp2<tab>website2-1#website2-2#website2-3...<tab>...
...
```

## Candidate Generation

The candidates of each query should also be preprocessed as a list in the file data/candidates.suggestion with the following json format.

```
...
{"query": <query>, "candidates": [<candidate1>, <candidate2>, <candidate3>, ...]}
...
```


## Heterogeneous Graph Construction and Embedding

TODO

## Training 

TODO

## Evalution

TODO

## Reference 

Jyun-Yu Jiang and Wei Wang. RIN: Reformulation Inference Network for Context-Aware Query Suggestion. In Proceedings of The 27th ACM International Conference on Information and Knowledge Management (CIKM '18), ACM, 2018.
