import torch
from torch_geometric.explain.config import ModelMode, ModelTaskLevel
from torch_geometric.utils import k_hop_subgraph, subgraph


def setup_models(models, device):
    res = []
    for name, model in models:
        res.append((name, model.to(device)))
    return res
