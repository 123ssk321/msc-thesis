import time

import torch
from torch_geometric.explain import DummyExplainer, Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain.config import ModelTaskLevel, ModelMode

from ciexplainer import CIExplainer
from link_pgexplainer import LinkPGExplainer

explainer_names = ['random_explainer', 'gnnexplainer', 'pgexplainer', 'subgraphx', 'cf_gnnexplainer', 'ciexplainer']


def get_explainer(explainer_name, explainer_config, model, model_config, threshold_config, **kwargs):
    if explainer_name == 'random_explainer':
        return get_random_explainer(explainer_config, model, model_config, threshold_config, **kwargs)
    if explainer_name == 'gnnexplainer':
        return get_gnnexplainer(explainer_config, model, model_config, threshold_config, **kwargs)
    if explainer_name == 'pgexplainer':
        return get_pgexplainer(explainer_config, model, model_config, threshold_config, **kwargs)
    # if explainer_name == 'cf_gnnexplainer':
    #     return get_cf_gnnexplainer(explainer_config, model, model_config, **kwargs)
    if explainer_name == 'ciexplainer':
        return get_ciexplainer(explainer_config, model, model_config, threshold_config, **kwargs)
    raise ValueError(f'Invalid explainer: {explainer_name}')


def get_random_explainer(explainer_config, model, model_config, threshold_config, **kwargs):
    explainer_algorithm = DummyExplainer().to(model.device)
    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        node_mask_type=explainer_config.node_mask_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
        threshold_config=threshold_config
    )
    return explainer


def get_gnnexplainer(explainer_config, model, model_config, threshold_config, **kwargs):
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
        threshold_config=threshold_config
    )
    return explainer


def get_pgexplainer(explainer_config, model, model_config, threshold_config, **kwargs):
    epochs = kwargs.get('epochs', 30)
    lr = kwargs.get('lr', 0.003)
    explainer_algorithm = LinkPGExplainer(epochs=epochs, lr=lr).to(model.device)

    explainer = Explainer(
        model=model,
        algorithm=explainer_algorithm,
        explanation_type=explainer_config.explanation_type,
        edge_mask_type=explainer_config.edge_mask_type,
        model_config=model_config,
        threshold_config=threshold_config
    )
    dataset = kwargs.get('dataset', None)
    if dataset is None:
        raise ValueError('Dataset is required for PGExplainer')
    # ls = []
    start_time = time.time()
    best_loss = float('inf')

    num_indices = 100
    edge_label_indices = kwargs.get('edge_label_indices', None)
    for epoch in range(epochs):
        loss = 0
        for idx in range(num_indices):  # Indices to train against.
            edge_label_idx = edge_label_indices[:, idx].view(-1, 1)
            # Explain a selected target (phenomenon) for a single edge:
            t = dataset.edge_label[idx].unsqueeze(dim=0).long()
            loss += explainer.algorithm.train(epoch, model, dataset.x, dataset.edge_index,
                                              target=t, edge_label_index=edge_label_idx, num_graphs=dataset.num_graphs)
        loss /= num_indices
        if loss < best_loss:
            best_loss = loss
    end_time = time.time()
    elapsed = (end_time - start_time) / 60
    print(f'PGExplainer took {elapsed:.2f} minutes to train. Best loss: {best_loss:.4f}')
    return explainer


def get_ciexplainer(explainer_config, model, model_config, threshold_config, **kwargs):
    dataset_name = kwargs.get('dataset_name', None)
    l = 11
    data = kwargs.get('dataset', None)
    device = model.device
    if model_config.task_level in [ModelTaskLevel.node, ModelTaskLevel.graph]:
        bin_feat_indices = kwargs.get('bin_feat_indices', None)
        cat_feat_indices = kwargs.get('cat_feat_indices', None)
        cont_feat_indices = kwargs.get('cont_feat_indices', None)
        if type(data) == list:
            data = data[0]
        # Explaining synthetic datasets
        if bin_feat_indices is None and cat_feat_indices is None and cont_feat_indices is None:
            bin_feat_indices = []
            cat_feat_indices = []
            cont_feat_indices = [i for i in range(data.x.size(1))]

        # Explaining mutag dataset
        if (cat_feat_indices is not None) and bin_feat_indices is None and cont_feat_indices is None:
            bin_feat_indices = []
            cat_feat_indices = cat_feat_indices
            cont_feat_indices = []

        features_metadata = {}
        for cont_feat_idx in cont_feat_indices:
            features_metadata[cont_feat_idx] = [min(data.x[:, cont_feat_idx]), max(data.x[:, cont_feat_idx]), 'float']

        for cat_feat_idx in cat_feat_indices:
            features_metadata[cat_feat_idx] = data.x.size(1)

    if model_config.task_level == ModelTaskLevel.edge:
        dataset_name = kwargs.get('dataset_name', None)
        bin_feat_indices = []
        cat_feat_indices = []
        cont_feat_indices = []
        features_metadata = {}
        if dataset_name == 'medref':
            age = 0
            cont_feat_indices.append(age)
            features_metadata[age] = [data.x[:, age].unique(), 'int']
            gender = data.x.size(1) - 1
            bin_feat_indices.append(gender)
            for cont_feat_idx in range(1, data.x.size(1) - 1):
                cont_feat_indices.append(cont_feat_idx)
                features_metadata[cont_feat_idx] = [torch.min(data.x[:, cont_feat_idx]),
                                                    torch.max(data.x[:, cont_feat_idx]), 'float']
        elif dataset_name == 'cora':
            start, end = 0, data.x.size(1)
            cat_feat_indices = [start]
            features_metadata[start] = end

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
        threshold_config=threshold_config
    )
    return explainer
