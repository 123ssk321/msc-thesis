
import random
import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Negative Edge Sampling function (parallelized)
def sample_negative_edges_df_for_dt(nodes, positive_edges_df_in_dt, date_time):
    positive_set = set(positive_edges_df_in_dt[["source", "target"]].itertuples(index=False, name=None))
    
    def valid_neg_edge(src, tgt):
        return (
            src != tgt
            and (src, tgt) not in positive_set
            and (tgt, src) not in positive_set
        )
    
    random_weight = np.random.uniform(positive_edges_df_in_dt['weight'].min(), positive_edges_df_in_dt['weight'].max())
    possible_neg_edges = [
        [src, tgt, random_weight, date_time] for src in nodes for tgt in nodes if valid_neg_edge(src, tgt)
    ]
    neg_edges = np.array(random.sample(possible_neg_edges, k=len(positive_set)))
    return {
        'source': neg_edges[:, 0].tolist(),
        'target': neg_edges[:, 1].tolist(),
        'weight': neg_edges[:, 2].tolist(),
        'timestamp': neg_edges[:, 3].tolist()
    }

# Parallelized Negative Edge Sampling
def sample_negative_edges_df(nodes, positive_edges_df, timestamps):
    def process_timestamp(i):
        date_time = timestamps[i]
        return sample_negative_edges_df_for_dt(nodes, positive_edges_df[positive_edges_df['timestamp'] == date_time], date_time)

    with Pool(cpu_count()) as pool:
        results = pool.map(process_timestamp, range(len(timestamps)))

    edges = {'source': [], 'target': [], 'weight': [], 'timestamp': []}
    for edges_data in results:
        edges['source'] += edges_data['source']
        edges['target'] += edges_data['target']
        edges['weight'] += edges_data['weight']
        edges['timestamp'] += edges_data['timestamp']

    edges_df = pd.DataFrame(edges)
    edges_df['source'] = edges_df['source'].astype(np.int64)
    edges_df['target'] = edges_df['target'].astype(np.int64)
    edges_df['weight'] = edges_df['weight'].astype(np.float64)
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    return edges_df

# Feature and Index conversion to PyTorch tensors (parallelized)
def feat_idx_w(num_nodes, node_index, nodes_df, edges_df):
    x = nodes_df.values[:, :].astype(float)
    row = edges_df['source'].map(node_index.get)
    col = edges_df['target'].map(node_index.get)
    data = [1] * len(edges_df)
    coo = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    edge_index = np.array([coo.row, coo.col], dtype=np.int64)
    edge_attr = edges_df['weight'].values.astype(float)
    return torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)

def to_feats_idxs_ws_parallel(timestamps, num_nodes, node_index, nodes_df, edges_df):
    def process_timestamp(ts):
        nodes_df_ts = nodes_df[nodes_df['TIMESTAMP'] == ts]
        edges_df_ts = edges_df[edges_df['timestamp'] == ts]
        return feat_idx_w(num_nodes, node_index, nodes_df_ts.drop('TIMESTAMP', axis=1), edges_df_ts)

    with Pool(cpu_count()) as pool:
        results = pool.map(process_timestamp, timestamps)

    features, edge_indices, edge_weights = zip(*results)
    return list(features), list(edge_indices), list(edge_weights)
