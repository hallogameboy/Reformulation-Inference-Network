# Reformulation-Inference-Network

## Required Packages

* Python 3.5.2 (or a compatible version)
* Tensorflow 1.4 (or a compatible version)
* NumPy
* SciPy
* Yaml
* ujson
* smart_open

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

The embeddings in the model should be generated by a graph embedding model.

### Graph Construction

Based on the preprocessed search logs, the script build_graph.py is available for generating the constructed graph.

### Graph Embedding

Based on the generated graph, any of graph embedding can be applied to embedding generation, e.g., node2vec. Then all of the mebeddings can be organized as the following format.

### Node ID Dictionary

In order to quickly lookup the ID of each node, a Python dictionary is required to be stored into a pickled file with the following format.

```
{...
('query', <query>}: <ID>,
('term', <term>}: <ID>,
{'site', <website>}: <ID>,
...}
```

### Embedding Format

All of the embeddings should be stored in a plain text file with the following format.

```
<number of nodes> <number or dimensions, N>
...
ID <feature 1> <feature 2> ... <feature N>
...
```

Note that IDs should be numerical values from 0 to N-1.

### Configuration

The paths of required files should be described in RIN/config.yml as follows.
More precisely, there are four required files/directories, including the node ID dictionary, the directory of preprocessed data, the embeddings, and the path to generate predictions.

```yaml
path_nodeid: <path_to_nodeid_dictionary>

path_data: <path_to_data>

path_emb: <path_to_embeddings>

path_prediction: <path_to_prediction>
```

## Training and Prediction

After the preparation of the above files, the model can be trained with the training sessions and predict the queries with the testing sessions. The command-lin instruction is as follows.

```bash
cd RIN
python3 train.py
```

More parameter settings can be looked up in the top of the code. In the training processing, the model will keep being trained with the training data and evaluated with the validation data (randomly separated in the code). While it reaches the best validation loss, the model will prediction the scores with testing sessions into the given path of predictions.


## Evalution

To evaluate the performance, eval.py has the exactly same format of parameter settings to train.py. The following command-line instruction can help evaluate the performance of the predictions in the given path.

```bash
cd RIN
python3 eval.py
```

## Reference 

Jyun-Yu Jiang and Wei Wang. RIN: Reformulation Inference Network for Context-Aware Query Suggestion. In Proceedings of The 27th ACM International Conference on Information and Knowledge Management (CIKM '18), ACM, 2018.
