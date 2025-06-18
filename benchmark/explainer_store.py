import time

import torch
from torch_geometric.explain import DummyExplainer, Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ModelTaskLevel, ModelMode

from ciexplainer import CIExplainer
from link_pgexplainer import LinkPGExplainer
from subgraphX_explainer import SubgraphXExplainer
from model_store import model_names

explainer_names = ['random_explainer', 'gnnexplainer', 'pgexplainer', 'subgraphx', 'ciexplainer']


def get_explainer(explainer_name, explainer_config, model, model_config, **kwargs):
    if explainer_name == 'random_explainer':
        return get_random_explainer(explainer_config, model, model_config, **kwargs)
    if explainer_name == 'gnnexplainer':
        return get_gnnexplainer(explainer_config, model, model_config, **kwargs)
    if explainer_name == 'pgexplainer':
        return get_pgexplainer(explainer_config, model, model_config, **kwargs)
    if explainer_name == 'subgraphx':
        return get_subgraphx(explainer_config, model, model_config, **kwargs)
    if explainer_name == 'ciexplainer':
        return get_ciexplainer(explainer_config, model, model_config, **kwargs)
    raise ValueError(f'Invalid explainer: {explainer_name}')


def get_random_explainer(explainer_config, model, model_config, **kwargs):
    explainer_algorithm = DummyExplainer().to(model.device)
    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        node_mask_type=explainer_config.node_mask_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
    )
    return explainer


def get_gnnexplainer(explainer_config, model, model_config, **kwargs):
    epochs = kwargs.get('epochs', 300)
    lr = kwargs.get('lr', 0.001)
    explainer_algorithm = GNNExplainer(epochs=epochs, lr=lr).to(model.device)
    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        node_mask_type=explainer_config.node_mask_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
    )
    return explainer


def get_pgexplainer(explainer_config, model, model_config, **kwargs):
    epochs = kwargs.get('epochs', 100)
    lr = kwargs.get('lr', 0.003)
    if model_config.task_level == ModelTaskLevel.edge:
        epochs = 30
        explainer_algorithm = LinkPGExplainer(epochs=epochs, lr=lr).to(model.device)
    else:
        explainer_algorithm = PGExplainer(epochs=epochs, lr=lr).to(model.device)

    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
    )
    dataset = kwargs.get('dataset', None)
    if dataset is None:
        raise ValueError('Dataset is required for PGExplainer')
    
    start_time = time.time()
    best_loss = float('inf')
    if model_config.task_level == ModelTaskLevel.node:
        num_indices = dataset.test_mask.nonzero().view(-1).size(0)
        for epoch in range(epochs):
            loss = 0
            for index in range(num_indices):
                loss += explainer.algorithm.train(epoch, model, dataset.x, dataset.edge_index, 
                                                  target=dataset.y, index=index)
            loss /= num_indices
            if loss < best_loss:
                best_loss = loss

    elif model_config.task_level == ModelTaskLevel.graph:
        num_indices = len(dataset)
        for epoch in range(epochs):
            loss = 0
            for data in dataset[:num_indices]:
                loss += explainer.algorithm.train(epoch, model, data.x, data.edge_index, target=data.y)
            loss /= num_indices
            if loss < best_loss:
                best_loss = loss

    elif model_config.task_level == ModelTaskLevel.edge:
        edge_label_indices = kwargs.get('edge_label_indices', None)
        num_indices = edge_label_indices.size(1)
        for epoch in range(epochs):
            loss = 0
            for idx in range(num_indices):
                edge_label_idx = edge_label_indices[:, idx].view(-1, 1)
                t = dataset.edge_label[idx].unsqueeze(dim=0).long()
                loss += explainer.algorithm.train(epoch, model, dataset.x, dataset.edge_index,
                                                  target=t, edge_label_index=edge_label_idx)
            loss /= num_indices
            if loss < best_loss:
                best_loss = loss
    else:
        raise ValueError(f'Invalid task level: {model_config.task_level}')
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f'PGExplainer took {elapsed:.2f} minutes to train. Best loss: {best_loss:.4f}')
    return explainer


def get_subgraphx(explainer_config, model, model_config, **kwargs):
    num_classes = model.model.out_channels
    if model_config.mode == ModelMode.binary_classification:
        num_classes += 1
    explainer_algorithm = SubgraphXExplainer(
        num_classes=num_classes, device=model.device, verbose=True, sample_num=10, rollout=10).to(
        model.device)
    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        node_mask_type=explainer_config.node_mask_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
    )
    return explainer


def get_ciexplainer(explainer_config, model, model_config, **kwargs):
    dataset_name = kwargs.get('dataset_name', None)
    data = kwargs.get('dataset', None)
    if dataset_name in ['ba_shapes', 'ba_shapes_link']:
        l = 700
    elif dataset_name in ['tree_grid', 'tree_grid_link']:
        l = 1231
    elif dataset_name == 'ba_2motif':
        l = 25
    else:#MUTAG
        l = data[0].x.size(0)
    
    device = model.device
    
    if type(data) == list:
        data = data[0]

    bin_feat_indices = []
    cat_feat_indices = []
    cont_feat_indices = []
    # Explaining synthetic datasets
    if dataset_name in ['ba_shapes', 'ba_shapes_link', 'tree_grid', 'tree_grid_link', 'ba_2motif']:
        cont_feat_indices = [i for i in range(data.x.size(1))]

    # Explaining mutag dataset
    if dataset_name == 'mutag':
        cat_feat_indices = [0]

    features_metadata = {}
    for cont_feat_idx in cont_feat_indices:
        features_metadata[cont_feat_idx] = [min(data.x[:, cont_feat_idx]), max(data.x[:, cont_feat_idx]), 'float']

    for cat_feat_idx in cat_feat_indices:
        features_metadata[cat_feat_idx] = data.x.size(1)

    explainer_algorithm = CIExplainer(
        l=l,
        bin_feat_indices=bin_feat_indices,
        cat_feat_indices=cat_feat_indices,
        cont_feat_indices=cont_feat_indices,
        features_metadata=features_metadata,
        tqdm_disable=True,
    ).to(device)
    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        node_mask_type=explainer_config.node_mask_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
    )
    return explainer
