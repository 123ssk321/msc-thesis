import argparse
import random
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm


def import_satellite_graph(**kwargs):
    reduced, frac1 = kwargs.get('options_reduced', (False, 0.25))
    reduced_sample_alt_e, frac2, min_alt, max_alt, e_thres, sampled1 = kwargs.get('options_sample_alt_e',
                                                                                  (False, 1.0, 500, 600, 0.2, False))
    reduced_sample_leos, frac3, leo, sampled2, = kwargs.get('options_sample_leos',
                                                            (True, 0.25, 'leo4', True))  # smallest LEO

    if reduced:
        nodes_savepath = f"datasets/space-track-ap2-graph-node-feats-reduced-{int(frac1 * 100)}.csv"
    elif reduced_sample_alt_e:
        if sampled1:
            nodes_savepath = f"datasets/space-track-ap2-graph-node-feats-reduced-{int(frac2 * 100)}-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
        else:
            nodes_savepath = f"datasets/space-track-ap2-graph-node-feats-reduced-h-{min_alt}-{max_alt}-e-{int(e_thres * 100)}.csv"
    elif reduced_sample_leos:
        if sampled2:
            nodes_savepath = f"datasets/space-track-ap2-graph-node-feats-{leo}-reduced-{int(frac3 * 100)}.csv"
        else:
            nodes_savepath = f"datasets/space-track-ap2-graph-node-feats-{leo}.csv"
    else:
        nodes_savepath = 'datasets/space-track-ap2-graph-node-feats.csv'
    nodes_savepath = kwargs.get('data_folder_location', '') + nodes_savepath

    nodes_df = pd.read_csv(nodes_savepath, memory_map=True).set_index('NORAD_CAT_ID').drop(
        ['OBJECT_NAME', 'OBJECT_ID', 'DECAY_DATE', 'CENTER_NAME', 'REF_FRAME', 'TIME_SYSTEM', 'MEAN_ELEMENT_THEORY',
         'EPHEMERIS_TYPE', 'CLASSIFICATION_TYPE'],
        axis=1).fillna({'CONSTELLATION_DISCOS_ID': 0})

    # ------------------------------------------------------ Normalize the node features -----------------------------------------------------
    scaler = MinMaxScaler()
    numeric_cols = nodes_df.select_dtypes(include=['float64', 'int64']).columns.drop('CONSTELLATION_DISCOS_ID')
    # Normalize only the numeric columns
    nodes_df[numeric_cols] = scaler.fit_transform(nodes_df[numeric_cols])

    # ------------------------------------------------------ One-hot encode categorical columns ---------------------------------------------
    nodes_df = pd.get_dummies(nodes_df, columns=['CONSTELLATION_DISCOS_ID', 'OBJECT_TYPE', 'RCS_SIZE'],
                              drop_first=False, dummy_na=True, dtype=float)

    edges_df = pd.read_csv(nodes_savepath.replace('node-feats', 'edges'), memory_map=True).rename(
        columns={'datetime': 'timestamp'})
    idx = kwargs.get('weight', -1)
    distances = ['r_dist', 'ct_dist', 'it_dist', 'dist']
    edges_df['weight'] = edges_df[distances[idx]]
    timestamps = edges_df['timestamp'].unique()
    return nodes_df, edges_df, timestamps


def temporal_signal_split(timestamps, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    train_snapshots = int(train_ratio * timestamps.shape[0])
    # necessary to reduce len(train_timestamps) timestamps to 1 such that there is only one feature matrix
    train_snapshots = train_snapshots - 1 if train_snapshots % 2 == 0 else train_snapshots
    val_snapshots = int(val_ratio * timestamps.shape[0])

    test_snapshots_start = train_snapshots + val_snapshots

    train_timestamps = timestamps[0:train_snapshots]
    val_timestamps = timestamps[train_snapshots:test_snapshots_start]
    test_timestamps = timestamps[test_snapshots_start:]

    return train_timestamps, val_timestamps, test_timestamps


def sample_negative_edges_df_for_dt(positive_edges_df_in_dt, date_time):
    s = positive_edges_df_in_dt['source'].unique()
    t = positive_edges_df_in_dt['target'].unique()
    nodes = np.union1d(s, t).tolist()
    positive_set = set(positive_edges_df_in_dt[["source", "target"]].itertuples(index=False, name=None))

    def valid_neg_edge(src, tgt):
        return (
            # no self-loops
                src != tgt
                and
                # neither direction of the edge should be a positive one
                (src, tgt) not in positive_set
                and (tgt, src) not in positive_set
        )

    random_weight = np.random.uniform(positive_edges_df_in_dt['weight'].min(), positive_edges_df_in_dt['weight'].max())
    possible_neg_edges = [
        [src, tgt, random_weight, date_time] for src in nodes for tgt in nodes if valid_neg_edge(src, tgt)
    ]
    neg_edges = np.array(random.sample(possible_neg_edges, k=len(positive_set)))
    return {'source': neg_edges[:, 0].tolist(), 'target': neg_edges[:, 1].tolist(), 'weight': neg_edges[:, 2].tolist(),
            'timestamp': neg_edges[:, 3].tolist()}


def sample_negative_edges_df(positive_edges_df, timestamps):
    edges = {'source': [], 'target': [], 'weight': [], 'timestamp': []}
    for i in tqdm(range(len(timestamps))):
        date_time = timestamps[i]
        edges_data = sample_negative_edges_df_for_dt(positive_edges_df[positive_edges_df['timestamp'] == date_time],
                                                     date_time)
        edges['source'] = edges['source'] + edges_data['source']
        edges['target'] = edges['target'] + edges_data['target']
        edges['weight'] = edges['weight'] + edges_data['weight']
        edges['timestamp'] = edges['timestamp'] + edges_data['timestamp']
    edges_df = pd.DataFrame(edges)
    edges_df['source'] = edges_df['source'].astype(np.int64)
    edges_df['target'] = edges_df['target'].astype(np.int64)
    edges_df['weight'] = edges_df['weight'].astype(np.float64)
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    return edges_df


def feat_idx_w(nodes_df, edges_df):
    nodes = list(nodes_df.index.unique())
    node_index = {node: i for i, node in enumerate(nodes)}
    x = nodes_df.values[:, :].astype(float)

    # Convert DataFrame to COO format
    row = edges_df['source'].map(node_index.get).to_numpy()
    col = edges_df['target'].map(node_index.get).to_numpy()
    src_tgt = np.stack((row, col), axis=0)
    tgt_src = np.stack((col, row), axis=0)
    edge_index = np.concatenate((src_tgt, tgt_src), axis=1) # undirected graph by PyG convention

    edge_attr = edges_df['weight'].values.astype(float)
    edge_attr = np.append(edge_attr, edge_attr)

    return torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attr)


def to_feats_idxs_ws(timestamps, nodes_df, edges_df):
    features = []
    edge_indices = []
    edge_weights = []
    for ts in tqdm(timestamps):
        nodes_df_ts = nodes_df[nodes_df['TIMESTAMP'] == ts]
        edges_df_ts = edges_df[edges_df['timestamp'] == ts]

        x, edge_index, edge_attr = feat_idx_w(nodes_df_ts.drop('TIMESTAMP', axis=1), edges_df_ts)
        features.append(x)
        edge_indices.append(edge_index)
        edge_weights.append(edge_attr)
    return features, edge_indices, edge_weights


def sample_negative_edges(pos_edges_df, timestamps):
    # This function will be called in a separate process for each dataset split
    return sample_negative_edges_df(pos_edges_df, timestamps)


def to_tensor_data(timestamps, nodes_df, edges_df):
    # This function will be called in a separate process for each dataset split
    return to_feats_idxs_ws(timestamps, nodes_df, edges_df)


def sample_new(data_folder_location, weight, **kwargs):
    save = True
    options_sample_leos = (True, 0.25, 'leo4', True)
    options_sample_alt_e = (True, 1.0, 500, 520, 0.25, False)
    nodes_df, edges_df, timestamps = import_satellite_graph(options_sample_alt_e=options_sample_alt_e,
                                                            data_folder_location=data_folder_location,
                                                            weight=weight)
    save_path = data_folder_location + 'datasets/'
    print(f'Saving to {save_path}')
    print(timestamps)
    print(timestamps.size)

    train_timestamps, val_timestamps, test_timestamps = temporal_signal_split(timestamps)

    train_edges_df = edges_df[edges_df['timestamp'].isin(train_timestamps)]
    val_edges_df = edges_df[edges_df['timestamp'].isin(val_timestamps)]
    test_edges_df = edges_df[edges_df['timestamp'].isin(test_timestamps)]

    print(
        f"Number of train timestamps: {train_timestamps.shape[0]}\n"
        f"Number of validation timestamps: {val_timestamps.shape[0]}\n"
        f"Number of test timestamps: {test_timestamps.shape[0]}"
    )

    # nodes = list(nodes_df.index.unique())
    # num_nodes = len(nodes)
    # node_index = {node: i for i, node in enumerate(nodes)}

    # Train set
    train_pos_edges_df = train_edges_df[['source', 'target', 'weight', 'timestamp']]

    print("Sampling negative training edges...")
    train_neg_edges_df = sample_negative_edges(train_pos_edges_df, train_timestamps)

    print("Converting to pyg tensor data...")
    train_pos_data = to_tensor_data(train_timestamps, nodes_df, train_pos_edges_df)
    train_neg_data = to_tensor_data(train_timestamps, nodes_df, train_neg_edges_df)

    if save:
        print("Saving...")
        torch.save(train_pos_data, f'{save_path}train_pos_data.pt')
        torch.save(train_neg_data, f'{save_path}train_neg_data.pt')

        np.save(f'{save_path}train_timestamps.npy', train_timestamps)

    print(
        f"Number of total positive edges in training set: {train_pos_edges_df.shape[0]}\n"
        f"Number of total negative edges in training set: {train_neg_edges_df.shape[0]}\n"
    )

    # Validation set
    val_pos_edges_df = val_edges_df[['source', 'target', 'weight', 'timestamp']]

    print("Sampling negative validation edges...")
    val_neg_edges_df = sample_negative_edges(val_pos_edges_df, val_timestamps)

    print("Converting to pyg tensor data...")
    val_pos_data = to_tensor_data(val_timestamps, nodes_df, val_pos_edges_df)
    val_neg_data = to_tensor_data(val_timestamps, nodes_df, val_neg_edges_df)

    if save:
        print("Saving...")
        torch.save(val_pos_data, f'{save_path}val_pos_data.pt')
        torch.save(val_neg_data, f'{save_path}val_neg_data.pt')

        np.save(f'{save_path}val_timestamps.npy', val_timestamps)

    print(
        f"Number of total positive edges in validation set: {val_pos_edges_df.shape[0]}\n"
        f"Number of total negative edges in validation set: {val_neg_edges_df.shape[0]}\n"
    )

    # Test set
    test_pos_edges_df = test_edges_df[['source', 'target', 'weight', 'timestamp']]

    print("Sampling negative testing edges...")
    test_neg_edges_df = sample_negative_edges(test_pos_edges_df, test_timestamps)

    print("Converting to pyg tensor data...")
    test_pos_data = to_tensor_data(test_timestamps, nodes_df, test_pos_edges_df)
    test_neg_data = to_tensor_data(test_timestamps, nodes_df, test_neg_edges_df)

    if save:
        print("Saving...")
        torch.save(test_pos_data, f'{save_path}test_pos_data.pt')
        torch.save(test_neg_data, f'{save_path}test_neg_data.pt')

        np.save(f'{save_path}test_timestamps.npy', test_timestamps)

    print(
        f"Number of total positive edges in test set: {test_pos_edges_df.shape[0]}\n"
        f"Number of total negative edges in test set: {test_neg_edges_df.shape[0]}\n"
    )

    return nodes_df, train_pos_data, train_neg_data, val_pos_data, val_neg_data, test_pos_data, test_neg_data


def load_existing(data_folder_location, weight, **kwargs):
    options_sample_leos = (True, 0.25, 'leo4', True)
    options_sample_alt_e = (True, 1.0, 500, 520, 0.25, False)
    nodes_df, edges_df, timestamps = import_satellite_graph(options_sample_alt_e=options_sample_alt_e,
                                                            data_folder_location=data_folder_location,
                                                            weight=weight)

    train_pos_data = torch.load(data_folder_location + 'datasets/train_pos_data.pt')
    train_neg_data = torch.load(data_folder_location + 'datasets/train_neg_data.pt')
    # train_timestamps = np.load(data_folder_location+'datasets/train_timestamps.npy')

    val_pos_data = torch.load(data_folder_location + 'datasets/val_pos_data.pt')
    val_neg_data = torch.load(data_folder_location + 'datasets/val_neg_data.pt')
    # val_timestamps = np.load(data_folder_location+'datasets/val_timestamps.npy')

    test_pos_data = torch.load(data_folder_location + 'datasets/test_pos_data.pt')
    test_neg_data = torch.load(data_folder_location + 'datasets/test_neg_data.pt')
    # test_timestamps = np.load(data_folder_location+'datasets/test_timestamps.npy')

    return nodes_df, train_pos_data, train_neg_data, val_pos_data, val_neg_data, test_pos_data, test_neg_data


def main():
    parser = argparse.ArgumentParser(
        description='Train a spatio-temporal graph neural network on the space-track dataset')
    parser.add_argument('--data_folder_location', type=str, default='/data/f.caldas/ssk/',
                        help='The location of the data folder')
    args = parser.parse_args()
    print(args)
    sample_new(args.data_folder_location, -1)


if __name__ == '__main__':
    main()
