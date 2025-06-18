import pickle as pkl
import numpy as np

import torch
import torch_geometric as pyg
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from sklearn.model_selection import train_test_split

from pre_process_MUTAG import get_graph_data

nc_datasets = ['ba_shapes', 'tree_grid']
gc_datasets = ['ba_2motif', 'mutag']
lp_datasets = ['cora']
normal_distribution = {'std': 0.1}


def get_nc_dataset(dataset_path, dataset_name, std=0.1, new=True, std_str=''):
    normal_distribution['std'] = std
    if dataset_name in nc_datasets:
        path = f'{dataset_path}{dataset_name}/'
        return nc_syn_pkl_to_pyg_data(f'{dataset_path}{dataset_name}', dataset_name,
                                      std=std_str, new=new)
    if dataset_name == 'all':
        datasets = []
        for dataset_name in nc_datasets:
            path = f'{dataset_path}{dataset_name}/'
            dataset = nc_syn_pkl_to_pyg_data(f'{path}{dataset_name}', dataset_name,
                                             std=std_str, new=new)
            datasets.append((dataset_name, dataset))
        return datasets


def get_gc_dataset(dataset_path, dataset_name, std=0.1, **kwargs):
    normal_distribution['std'] = std
    if dataset_name in gc_datasets:
        path = f'{dataset_path}{dataset_name}/'
        return gc_pkl_to_pyg_data(f'{path}{dataset_name}', dataset_name, **kwargs)
    if dataset_name == 'all':
        datasets = []
        for dataset_name in gc_datasets:
            path = f'{dataset_path}{dataset_name}/'
            dataset, num_classes = gc_pkl_to_pyg_data(f'{path}{dataset_name}', dataset_name,
                                                       **kwargs)
            datasets.append((dataset_name, dataset, num_classes))
        return datasets


def get_lp_dataset(dataset_path, dataset_name):
    if dataset_name in lp_datasets:
        path = f'{dataset_path}{dataset_name}/'
        return lp_pth_to_pyg_data(f'{path}{dataset_name}', dataset_name)
    if dataset_name == 'all':
        datasets = []
        for dataset_name in lp_datasets:
            path = f'{dataset_path}{dataset_name}/'
            train_data, val_data, test_data = lp_pth_to_pyg_data(f'{path}{dataset_name}', dataset_name)
            datasets.append((dataset_name, train_data, val_data, test_data))
        return datasets


def generate_class_means(num_classes):
    """
    Generates a dictionary of mean values [X, Y] for each class.
    The means are arranged in a circular clockwise pattern on a 2D axis.

    Args:
        num_classes (int): Number of classes to generate means for.

    Returns:
        dict: A dictionary where keys are class indices and values are mean tensors.
    """
    class_means = {}

    # Directions to move: up, right, down, left (clockwise)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x, y = 0, 0  # Starting point
    steps_remaining = 1  # Steps to move in the current direction
    step_increase_count = 0  # To track when to increase step size
    direction_index = 0  # Start with moving up

    for i in range(num_classes):
        class_means[i] = torch.tensor([float(x), float(y)])

        # Move in the current direction
        dx, dy = directions[direction_index]
        x += dx
        y += dy
        steps_remaining -= 1

        # If no steps remain in the current direction, switch direction
        if steps_remaining == 0:
            direction_index = (direction_index + 1) % 4  # Rotate to next direction
            step_increase_count += 1

            # Increase steps every two direction changes
            if step_increase_count % 2 == 0:
                steps_remaining = (step_increase_count // 2) + 1
            else:
                steps_remaining = step_increase_count // 2 + 1

    # return class_means
    if num_classes == 4:
        return {0: torch.tensor([float(0), float(1)]),
                1: torch.tensor([float(1), float(0)]),
                2: torch.tensor([float(-1), float(0)]),
                3: torch.tensor([float(0), float(-1)])}
    else:
        return {0: torch.tensor([float(0), float(1)]), 1: torch.tensor([float(-1), float(0)])}


def combine_split_labels(y_train, y_val, y_test):
    """
    Combines train, validation, and test labels into a single array indicating the class for each node.

    Args:
        y_train (np.ndarray): Train set array of shape (num_nodes, num_classes).
        y_val (np.ndarray): Validation set array of shape (num_nodes, num_classes).
        y_test (np.ndarray): Test set array of shape (num_nodes, num_classes).

    Returns:
        np.ndarray: Combined array of shape (num_nodes,) where each element indicates the class for the corresponding node.
    """
    # Initialize an array to store the class for each node
    combined_classes = np.full(y_train.shape[0], -1, dtype=int)  # -1 for unclassified nodes

    # Check each dataset for class membership and assign class index
    num_classes = y_train.shape[1]
    for class_index in range(num_classes):
        combined_classes[y_train[:, class_index] == 1] = class_index
        combined_classes[y_val[:, class_index] == 1] = class_index
        combined_classes[y_test[:, class_index] == 1] = class_index

    return combined_classes


def nc_normal_feature_matrix(data):
    # Determine the number of classes
    num_classes = data.num_classes

    # Define mean values [X, Y] for each class
    class_means = generate_class_means(num_classes)

    # Initialize feature matrix
    feature_matrix = torch.empty((data.num_nodes, 2))

    # Assign features based on node classes
    for node_idx in range(data.num_nodes):
        node_class = data.y[node_idx].item()
        mean = class_means[node_class]  # Get mean [X, Y] for the class
        feature = torch.normal(mean=mean, std=normal_distribution['std'], out=torch.empty((1, 2)))
        feature_matrix[node_idx] = feature

    return feature_matrix


def ba2_normal_feature_matrix(data):
    # Define mean values [X, Y] for each class
    graph_class = data.y

    # house motif
    if graph_class == 0:
        class_means = {0: torch.tensor([float(0), float(1)]),
                       1: torch.tensor([float(1), float(0)]),
                       2: torch.tensor([float(-1), float(0)]),
                       3: torch.tensor([float(0), float(-1)])}
    # cycle motif
    else:
        class_means = {0: torch.tensor([float(1), float(1)]),
                       1: torch.tensor([float(-1), float(-1)]),
                       2: torch.tensor([float(-1), float(1)]),
                       3: torch.tensor([float(1), float(-1)])}

    # Initialize feature matrix
    feature_matrix = torch.empty((data.num_nodes, 2))

    # Assign features based on node classes
    for node_idx in range(data.num_nodes):
        if node_idx == 20:
            node_class = 1
        elif (node_idx == 21) or (node_idx == 24):
            node_class = 2
        elif (node_idx == 22) or (node_idx == 23):
            node_class = 3
        else:
            node_class = 0
        mean = class_means[node_class]  # Get mean [X, Y] for the class
        feature = torch.normal(mean=mean, std=normal_distribution['std'], out=torch.empty((1, 2)))
        feature_matrix[node_idx] = feature

    return feature_matrix


def nc_syn_pkl_to_pyg_data(syn_dataset_path, dataset_name, normal_features=True, std=None, new=True):
    """
    Converts a synthetic node classification dataset from a pickle file to a PyG Data object.

    Args:
        syn_dataset_path (str): Path to the synthetic dataset pickle file.
        dataset_name (str): Name of the synthetic dataset.
        normal_features (bool): Whether to assign a feature vector sampled from a 2d Gaussian distribution centered around a point dependent of the class of the node.

    Returns:
        pyg.data.Data: PyG Data object containing the dataset.
    """
    if std is not None and not new:
        data = torch.load(f'{syn_dataset_path}{std}.pth')
        print('$' * 101)
        print(f'Loading {dataset_name} dataset with {std} standard deviation from {syn_dataset_path}{std}.pth')
        print(f'Number of PyTorch Geometric Data object (undirected) edges: {data.num_edges}')
        print(f'Used feature matrix shape: {data.x.shape}')
        print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
        print(f'Number of ground truth edges: {data.gt_edges.shape[1]}')
        print(f'Node mask shape: {data.node_mask.shape}')
        print(f'Edge mask shape: {data.edge_mask.shape}')

        return data

    with open(f'{syn_dataset_path}.pkl', 'rb') as f:
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = pkl.load(f)

    print('$' * 101)
    print(f'{dataset_name} pickle file information')
    print(f'Adjacency matrix shape: {adj.shape}')
    print(f'Features matrix shape: {features.shape}')
    print(f'Train labels shape: {y_train.shape}')
    print(f'Validation labels shape: {y_val.shape}')
    print(f'Test labels shape: {y_test.shape}')
    print(f'Train mask shape: {train_mask.shape}')
    print(f'Validation mask shape: {val_mask.shape}')
    print(f'Test mask shape: {test_mask.shape}')
    print(f'Number of nodes: {features.shape[0]}')
    print(f'Number of training nodes: {train_mask[train_mask == True].shape[0]}')
    print(f'Number of validation nodes: {val_mask[val_mask == True].shape[0]}')
    print(f'Number of test nodes: {test_mask[test_mask == True].shape[0]}')
    print(f'Number of classes: {y_train.shape[1]}')
    print(f'Number of edges: {adj[adj == 1].shape[0]}')
    from torch_geometric.utils import dense_to_sparse
    y = combine_split_labels(y_train, y_val, y_test)
    edge_index = dense_to_sparse(torch.tensor(adj))[0].numpy()  # np.vstack(np.nonzero(adj))
    print(f': {edge_index.shape}')
    gt_edges = np.vstack(np.nonzero(edge_label_matrix))

    # expand dims, broadcast, align shapes, reduce dims(all), reduce dims(any)
    edge_mask = np.any(np.all(edge_index.T[:, None, :] == gt_edges.T[None, :, :], axis=2), axis=1)
    node_mask = np.zeros(features.shape[0], dtype=int)
    unique_nodes = np.unique(gt_edges)
    node_mask[unique_nodes] = 1
    data = pyg.data.Data(
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool),
        gt_edges=torch.tensor(gt_edges, dtype=torch.long),
        edge_mask=torch.tensor(edge_mask, dtype=torch.int32),
        node_mask=torch.tensor(node_mask, dtype=torch.int32)
    )
    data.num_classes = y_train.shape[1]
    data.num_nodes = y_train.shape[0]
    if normal_features:
        data.x = nc_normal_feature_matrix(data)
    else:
        data.x = torch.tensor(features, dtype=torch.float)
    print(f'Number of PyTorch Geometric Data object (undirected) edges: {data.num_edges}')
    print(f'Used feature matrix shape: {data.x.shape}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Number of ground truth edges: {data.gt_edges.shape[1]}')
    print(f'Number of ground truth nodes: {unique_nodes.shape[0]}')
    print(f'Node mask shape: {data.node_mask.shape}')
    print(f'Edge mask shape: {data.edge_mask.shape}')

    return data


def gc_pkl_to_pyg_data(syn_dataset_path, dataset_name, normal_features=True, **kwargs):
    explain = kwargs.get('explain', False)
    std_str = kwargs.get('std_str', None)
    if explain and ('mu' in dataset_name):
        return explain_mutag_pkl_to_pyg_data(syn_dataset_path, dataset_name)
    elif std_str is not None:
        print(f'Loading {dataset_name} dataset with {std_str} standard deviation from {syn_dataset_path}{std_str}.pth')
        data_list = torch.load(f'{syn_dataset_path}{std_str}.pth')
        num_classes = 2
        return data_list, num_classes
    else:
        with open(syn_dataset_path, 'rb') as f:
            adj, x, y = pkl.load(f)

        print('$' * 101)
        print(f'{dataset_name} pickle file information')
        print(f'Adjacency matrix shape: {adj.shape}')
        print(f'Features matrix shape: {x.shape}')
        print(f'Labels shape: {y.shape}')
        print(f'Number of graphs: {x.shape[0]}')
        num_classes = int(y.max().item()) + 1
        print(f'Number of classes: {num_classes}')
        print(f'Number of nodes: {x.shape[1]}')
        print(f'Number of edges: {int(adj.sum())}')

        xs = torch.from_numpy(x).to(torch.float)
        ys = torch.from_numpy(y).argmax(dim=-1).to(torch.long)

        data_list = []
        for i in range(xs.size(0)):
            row_indices, col_indices = np.nonzero(adj[i])
            # Undirected graph: Add edges in both directions
            edge_index = np.vstack((row_indices, col_indices))

            x = xs[i]
            y = ys[i]
            data = pyg.data.Data(edge_index=torch.tensor(edge_index, dtype=torch.long), y=y)
            data.num_classes = num_classes
            data.num_nodes = x.size(0)
            if normal_features:
                data.x = ba2_normal_feature_matrix(data)
            else:
                data.x = x

            data_list.append(data)
        return data_list, num_classes


def explain_mutag_pkl_to_pyg_data(syn_dataset_path, dataset_name):
    edge_lists, graph_labels, edge_label_lists, node_label_lists = get_graph_data(dataset_name)
    with open(f'{syn_dataset_path}.pkl', 'rb') as f:
        adj, x, y = pkl.load(f)

    # only consider the mutagen graphs with NO2 and NH2.
    selected = []
    for gid in range(adj.shape[0]):
        if np.argmax(y[gid]) == 0 and np.sum(edge_label_lists[gid]) > 0:
            selected.append(gid)

    print('Number of mutagen graphs with NO2 and NH2', len(selected))
    adj = adj[selected]
    x = x[selected]
    y = y[selected]
    edge_lists = [edge_lists[i] for i in selected]
    edge_label_lists = [edge_label_lists[i] for i in selected]
    node_label_lists = [node_label_lists[i] for i in selected]

    print('$' * 101)
    print(f'{dataset_name} pickle file information')
    print(f'Adjacency matrix shape: {adj.shape}')
    print(f'Features matrix shape: {x.shape}')
    print(f'Labels shape: {y.shape}')
    print(f'Number of graphs: {x.shape[0]}')
    num_classes = int(y.max().item()) + 1
    print(f'Number of classes: {num_classes}')
    print(f'Number of nodes: {x.shape[1]}')
    print(f'Number of edges: {int(adj.sum())}')

    xs = torch.from_numpy(x).to(torch.float)
    ys = torch.from_numpy(y).argmax(dim=-1).to(torch.long)
    num_features = xs.size(2)
    data_list = []
    for i in range(xs.size(0)):
        edge_index = torch.tensor(edge_lists[i], dtype=torch.long).T
        y = ys[i].reshape(1)
        x = torch.nn.functional.one_hot(torch.tensor(node_label_lists[i], dtype=torch.long),
                                        num_classes=num_features).float()
        edge_label = torch.tensor(edge_label_lists[i]).bool()
        data = pyg.data.Data(x=x, edge_index=edge_index, y=y, edge_mask=edge_label)
        data.num_classes = num_classes
        data.num_nodes = x.size(0)
        data.x = x

        data_list.append(data)
    return data_list, num_classes


def split_edges_stratified(edge_index, edge_label, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    # Ensure that train + val + test = 1
    assert train_ratio + val_ratio + test_ratio == 1, "The split ratios must sum to 1"

    # Get the number of edges
    num_edges = edge_label.shape[0]

    # First, split into train and temp (val + test) using stratification
    train_mask, temp_mask = train_test_split(torch.arange(num_edges), train_size=train_ratio,
                                             stratify=edge_label, random_state=random_seed)

    # Then split the temp into val and test using stratification
    val_mask, test_mask = train_test_split(temp_mask, test_size=test_ratio / (val_ratio + test_ratio),
                                           stratify=edge_label[temp_mask], random_state=random_seed)

    # Use the masks to index into the original edge_index and edge_label tensors
    train_edge_index = edge_index[:, train_mask]
    val_edge_index = edge_index[:, val_mask]
    test_edge_index = edge_index[:, test_mask]

    train_edge_label = edge_label[train_mask]
    val_edge_label = edge_label[val_mask]
    test_edge_label = edge_label[test_mask]

    return (train_edge_index, train_edge_label), (val_edge_index, val_edge_label), (test_edge_index, test_edge_label)


def lp_pth_to_pyg_data(dataset_path, dataset_name):
    train_data = torch.load(f'{dataset_path}_train_data.pth')
    val_data = torch.load(f'{dataset_path}_val_data.pth')
    test_data = torch.load(f'{dataset_path}_test_data.pth')

    print('$' * 101)
    print(f'{dataset_name} pickle file information')
    print(f'Train data: {train_data}')
    print(f'Validation data: {val_data}')
    print(f'Test data: {test_data}')

    return train_data, val_data, test_data
