from torch import Tensor
from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.config import ModelTaskLevel, ModelReturnType, ModelMode
from torch_geometric.typing import NodeType, EdgeType
from typing import Union, Dict, Optional

import torch.nn
import numpy as np
import torch_geometric as pyg
from torch_geometric.utils import k_hop_subgraph, subgraph
from tqdm import tqdm


class CIExplainer(pyg.explain.ExplainerAlgorithm):

    def __init__(self, l: int, bin_feat_indices: list, cat_feat_indices: list,
                 cont_feat_indices: list, features_metadata: dict, tqdm_disable=False):
        """

        :param l:
        :param bin_feat_indices:
        :param cat_feat_indices:
        :param cont_feat_indices:
        :param features_metadata:
        :param tqdm_disable:
        """
        super().__init__()
        self.l = l
        self.bin_feat_indices = bin_feat_indices
        self.cat_feat_indices = cat_feat_indices
        self.cont_feat_indices = cont_feat_indices
        self.features_metadata = features_metadata
        self.tqdm_disable = tqdm_disable

        self.node_mask = None
        self.edge_mask = None

    def forward(self, model: torch.nn.Module, x: Union[Tensor, Dict[NodeType, Tensor]],
                edge_index: Union[Tensor, Dict[EdgeType, Tensor]], *, target: Tensor,
                index: Optional[Union[int, Tensor]] = None, **kwargs) -> Union[Explanation, HeteroExplanation]:
        sub_mask = None
        sub_nodes = None
        if self.model_config.task_level == ModelTaskLevel.node:
            num_hops = model.model.num_layers
            sub_x, sub_edge_index, new_idx, sub_mask, sub_nodes = self._get_neighbourhood(index, num_hops, x, edge_index)
            if 'edge_weight' in kwargs:
                edge_weight = kwargs['edge_weight']
                sub_edge_weight = edge_weight[sub_mask]
                kwargs['edge_weight'] = sub_edge_weight

            neighborhood = (sub_x, sub_edge_index, new_idx, sub_mask)
        elif self.model_config.task_level == ModelTaskLevel.graph:
            neighborhood = (x, edge_index)
        elif self.model_config.task_level == ModelTaskLevel.edge:
            edge_label_index = kwargs.get('edge_label_index', None)
            if edge_label_index is None:
                raise ValueError("edge_label_index must be provided for edge-level tasks.")
            num_hops = model.model.num_layers
            sub_x, sub_edge_index, new_edge_label_index, sub_mask, sub_nodes = self._get_neighbourhood(edge_label_index.view(2), num_hops, x,
                                                                                           edge_index)
            if 'edge_weight' in kwargs:
                edge_weight = kwargs['edge_weight']
                sub_edge_weight = edge_weight[sub_mask]
                kwargs['edge_weight'] = sub_edge_weight

            neighborhood = (sub_x, sub_edge_index, new_edge_label_index.view(-1, 1), sub_mask)
        y = target
        if index is not None:
            y = target[index]

        topl_nodes, sub_edges = self._explain(model, y, neighborhood)
        # self.edge_mask = torch.zeros(edge_index.shape[1], device=edge_index.device)
        # self.node_mask = torch.zeros(x.shape[0], 1, device=x.device)
        # if sub_mask is not None and sub_nodes is not None:
        #     self.edge_mask[sub_mask] = sub_edges
        #     self.node_mask[sub_nodes] = topl_nodes
        # else:
        #     self.edge_mask = sub_edges
        #     self.node_mask = topl_nodes
        if sub_mask is not None and sub_nodes is not None:
            self.edge_mask = torch.zeros(edge_index.shape[1], device=edge_index.device, dtype=sub_edges.dtype)
            self.node_mask = torch.zeros((x.shape[0], 1), device=x.device, dtype=topl_nodes.dtype)
            self.edge_mask[sub_mask] = sub_edges
            self.node_mask[sub_nodes] = topl_nodes
        else:
            self.edge_mask = sub_edges
            self.node_mask = topl_nodes
        return Explanation(node_mask=self.node_mask, edge_mask=self.edge_mask)

    def supports(self) -> bool:
        return True

    def _get_neighbourhood(self, node_idx, num_hops, x, edge_index):
        sub_nodes, sub_edge_index, new_node_idx, sub_mask = k_hop_subgraph(node_idx, num_hops, edge_index,
                                                                           relabel_nodes=True)
        sub_x = x[sub_nodes]
        return sub_x, sub_edge_index, new_node_idx, sub_mask, sub_nodes

    def _explain(self, model: torch.nn.Module, pred_prob: Tensor, neighborhood: tuple):
        # x, edge_index = neighborhood[0], neighborhood[1]
        # node_mask = torch.zeros(x.size(0), 1, device=x.device)
        # #print(x.size(0))
        # for node, features in tqdm(enumerate(x), disable=self.tqdm_disable):
        #     causal_effect = self._causal_effect(model, pred_prob, features, neighborhood)
        #     node_mask[node] = causal_effect
        # k = self.l if self.l < x.size(0) else x.size(0)
        # topl_values, topl_nodes = torch.topk(node_mask[:, 0], k)
        # node_mask = torch.zeros(x.size(0), 1, device=x.device)
        # node_mask[topl_nodes] = topl_values.view(-1, 1)
        # sub_edges = torch.zeros(edge_index.size(1), device=edge_index.device)
        # _, _ , sub_mask = subgraph(topl_nodes, edge_index, return_edge_mask=True)
        # sub_edges[sub_mask] = 1.0
        # return node_mask, sub_edges
        x, edge_index = neighborhood[0], neighborhood[1]
        device = x.device

        node_mask = torch.zeros(x.size(0), 1, device=device)  # Initial allocation for node_mask

        # Loop through nodes, calculate causal effect and assign it to the node_mask in-place
        nodes = edge_index.reshape(-1).unique()
        for node in tqdm(nodes, disable=self.tqdm_disable):
            features = x[node]
            causal_effect = self._causal_effect(model, pred_prob, features, neighborhood)
            node_mask[node] = causal_effect  # In-place assignment

        k = min(self.l, x.size(0))  # Optimized `k` calculation

        # Use top-k values and nodes efficiently
        topl_values, topl_nodes = torch.topk(node_mask[:, 0], k)

        # Reset the node_mask tensor without reallocation
        node_mask.zero_()  # In-place reset to zero
        node_mask[topl_nodes] = topl_values.view(-1, 1)  # In-place assignment for top-k values

        # Preallocate sub_edges tensor
        sub_edges = torch.zeros(edge_index.size(1), device=edge_index.device)

        # Efficient subgraph extraction
        _, _, sub_mask = subgraph(topl_nodes, edge_index, return_edge_mask=True)
        sub_edges[sub_mask] = 1.0  # In-place assignment for subgraph edges

        return node_mask, sub_edges

    def _get_prediction(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, **kwargs):
        if 'edge_label_index' in kwargs:
            return model(x, edge_index, edge_label_index=kwargs['edge_label_index'])
        if 'index' in kwargs:
            return model(x, edge_index)[kwargs['index']]
        return model(x, edge_index)

    def _causal_effect(self, model: torch.nn.Module, pred_prob: Tensor, features: torch.Tensor, neighborhood: tuple):
        causal_effect = 0.0
        for bin_feat_idx in self.bin_feat_indices:
            counterfactual_pred_prob = self._feature_causal_effect(model, bin_feat_idx, "bin", features, neighborhood)

            feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)


            causal_effect = max(causal_effect, feature_causal_effect)

        for cat_feat_idx in self.cat_feat_indices:
            counterfactual_pred_prob = self._feature_causal_effect(model, cat_feat_idx, "cat", features, neighborhood)

            feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)
            causal_effect = max(causal_effect, feature_causal_effect)

        for cont_feat_idx in self.cont_feat_indices:
            counterfactual_pred_prob = self._feature_causal_effect(model, cont_feat_idx, "cont", features, neighborhood)

            feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)
            causal_effect = max(causal_effect, feature_causal_effect)

        # for bin_feat_idx in self.bin_feat_indices:
        #     original_value = features[bin_feat_idx]
        #     self._bin_counterfactual(bin_feat_idx, features)
        #     counterfactual_pred_prob = model(x, edge_index, edge_label_index, edge_weight=edge_weight)
        #     self._reset_features(bin_feat_idx, original_value, features)
        #
        #     feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)
        #     causal_effect = max(causal_effect, feature_causal_effect)
        #
        # for cat_feat_idx in self.cat_feat_indices:
        #     original_value = features[cat_feat_idx]
        #     self._cat_counterfactual(cat_feat_idx, features, original_value)
        #     counterfactual_pred_prob = model(x, edge_index, edge_label_index, edge_weight=edge_weight)
        #     self._reset_features(cat_feat_idx, original_value, features)
        #
        #     feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)
        #     causal_effect = max(causal_effect, feature_causal_effect)
        #
        # for cont_feat_idx in self.cont_feat_indices:
        #     original_value = features[cont_feat_idx]
        #     self._cont_counterfactual(cont_feat_idx, features, original_value)
        #     counterfactual_pred_prob = model(x, edge_index, edge_label_index, edge_weight=edge_weight)
        #     self._reset_features(cont_feat_idx, original_value, features)
        #
        #     feature_causal_effect = abs(counterfactual_pred_prob - pred_prob)
        #     causal_effect = max(causal_effect, feature_causal_effect)
        #print(causal_effect)
        return causal_effect

    def _feature_causal_effect(self, model: torch.nn.Module, feat_idx: int, feat_type: str, features: torch.Tensor,
                               neighborhood: tuple):

        original_value = features[feat_idx]
        if feat_type == "bin":
            self._bin_counterfactual(feat_idx, features)
        elif feat_type == "cat":
            original_value = torch.argmax(features).item()
            self._cat_counterfactual(feat_idx, features, original_value)
        elif feat_type == "cont":
            self._cont_counterfactual(feat_idx, features, original_value, model.device)
        else:
            raise ValueError(f"Unknown feature type: {feat_type}")

        if self.model_config.task_level == ModelTaskLevel.node:
            x, edge_index, node_idx, edge_weight = neighborhood
            output = model(x, edge_index)[node_idx]

            if self.model_config.mode == ModelMode.binary_classification:
                output = torch.sigmoid(output)

            if self.model_config.mode == ModelMode.multiclass_classification:
                output = torch.softmax(output, dim=-1)
                output = torch.max(output)

            counterfactual_pred_prob = output
        elif self.model_config.task_level == ModelTaskLevel.graph:
            x, edge_index = neighborhood
            output = model(x, edge_index)
            if self.model_config.mode == ModelMode.binary_classification:
                output = torch.sigmoid(output)
            counterfactual_pred_prob = output
        elif self.model_config.task_level == ModelTaskLevel.edge:
            x, edge_index, edge_label_index, edge_weight = neighborhood
            output = model(x, edge_index, edge_label_index=edge_label_index)
            if self.model_config.mode == ModelMode.binary_classification:
                output = torch.sigmoid(output)
            counterfactual_pred_prob = output

        self._reset_features(feat_idx, original_value, features, feat_type)
        #print(counterfactual_pred_prob)
        return counterfactual_pred_prob

    def _bin_counterfactual(self, bin_feat_idx: int, features: torch.Tensor):
        features[bin_feat_idx] = 1 - features[bin_feat_idx]

    def _cat_counterfactual(self, cat_feat_idx: int, features: torch.Tensor, original_value):
        # cat_feat_values = self.features_metadata[cat_feat_idx]
        # filtered_set = cat_feat_values[cat_feat_values != original_value]
        # features[cat_feat_idx] = np.random.choice(filtered_set)
        # Get current category index
        current_idx = torch.argmax(features).item()

        # Generate a new random index different from the current one
        start, end = cat_feat_idx, self.features_metadata[cat_feat_idx]
        valid_indices = list(range(start, end))
        valid_indices.remove(current_idx)
        new_idx = np.random.choice(valid_indices)
        # Update the features directly without creating a new tensor
        features.zero_()  # In-place operation to zero out the tensor
        features[new_idx] = 1.0

    def _cont_counterfactual(self, cont_feat_idx: int, features: torch.Tensor, original_value, device):
        feat_type = self.features_metadata[cont_feat_idx][-1]
        if feat_type == "int":
            values, _ = self.features_metadata[cont_feat_idx]
            # sampled = original_value
            # while sampled == original_value:
            #     sampled = values[torch.randint(0, values.size(0), (1,)).item()]
            # features[cont_feat_idx] = sampled
            # Remove the original value from the tensor and then randomly choose a new value
            values = values[values != original_value]  # Remove the original value
            sampled = values[torch.randint(0, values.size(0), (1,)).item()]  # Choose a new value

            # Re-add the original value to the tensor
            values = torch.cat([values, original_value.unsqueeze(0)])  # Keep it on the same device if necessary

            # Update the feature
            features[cont_feat_idx] = sampled
        else:
            min_val, max_val, _ = self.features_metadata[cont_feat_idx]
            sampled = original_value
            while sampled == original_value:
                sampled = torch.tensor(np.random.uniform(min_val.cpu(), max_val.cpu()), device=device)
            features[cont_feat_idx] = (sampled - min_val) / (max_val - min_val)

    def _reset_features(self, idx: int, original_value: float, features: torch.Tensor, feat_type: str):
        if feat_type == "cat":
            features.zero_()
            features[original_value] = 1.0
        else:
            features[idx] = original_value

    def _generate_counterfactual(self):
        pass
