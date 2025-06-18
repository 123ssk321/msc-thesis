# import math
# from typing import Any
#
# import torch
# from torch.utils.data import DataLoader
# from tsl.data import Data as TData
#
#
# class DTDGData(TData):
#     def __init__(self, input, target, edge_index, edge_weight, num_nodes):
#         super(DTDGData, self).__init__(input=input, target=target, edge_index=edge_index, edge_weight=edge_weight)
#         self.num_nodes = num_nodes
#
#     # def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
#     #     pass
#
#
# class DTDGLinkDataset(torch.utils.data.Dataset):
#     def __init__(self, pos_data, neg_data, window=11, horizon=1, stride=1):
#         self.pos_xs, self.pos_edge_idxs, self.pos_edge_attrs = pos_data
#         self.neg_xs, self.neg_edge_idxs, self.neg_edge_attrs = neg_data
#         self.num_nodes = self.pos_xs[0].size()[0]
#         self.window = window
#         self.horizon = horizon
#         self.stride = stride
#         self.tot_samples = math.ceil((len(self.pos_xs) - (self.window + self.horizon)) / self.stride)
#
#     def __len__(self):
#         return self.tot_samples
#
#     def __getitem__(self, idx):
#         start = idx * self.stride
#         end = start + self.window + self.horizon
#
#         sample_pos_xs, sample_pos_edge_idxs, sample_pos_edge_attrs = self.pos_xs[start: end], self.pos_edge_idxs[
#                                                                                               start: end], self.pos_edge_attrs[
#                                                                                                            start: end]
#         sample_neg_xs, sample_neg_edge_idxs, sample_neg_edge_attrs = self.neg_xs[start: end], self.neg_edge_idxs[
#                                                                                               start: end], self.neg_edge_attrs[
#                                                                                                            start: end]
#
#         window_pos_xs, window_pos_edge_idxs, window_pos_edge_attrs = sample_pos_xs[:self.window], sample_pos_edge_idxs[
#                                                                                                   :self.window], sample_pos_edge_attrs[
#                                                                                                                  :self.window]
#         window_neg_xs, window_neg_edge_idxs, window_neg_edge_attrs = sample_neg_xs[:self.window], sample_neg_edge_idxs[
#                                                                                                   :self.window], sample_neg_edge_attrs[
#                                                                                                                  :self.window]
#
#         target_pos_xs, target_pos_edge_idxs, target_pos_edge_attrs = sample_pos_xs[-1], sample_pos_edge_idxs[-1], \
#             sample_pos_edge_attrs[-1]
#         target_neg_xs, target_neg_edge_idxs, target_neg_edge_attrs = sample_neg_xs[-1], sample_neg_edge_idxs[-1], \
#             sample_neg_edge_attrs[-1]
#         input = dict(
#             pos_x=torch.cat(window_pos_xs, dim=0),
#             neg_x=torch.stack(window_neg_xs)
#         )
#         target = dict(
#             pos_y=torch.stack(target_pos_xs) if self.horizon > 1 else torch.stack([target_pos_xs]),
#             neg_y=torch.stack(target_neg_xs) if self.horizon > 1 else torch.stack([target_neg_xs])
#         )
#         edge_index = dict(
#             input_pos_ei=window_pos_edge_idxs,
#             input_neg_ei=window_neg_edge_idxs,
#             target_pos_ei=target_pos_edge_idxs,
#             target_neg_ei=target_neg_edge_idxs,
#         )
#         edge_weight = dict(
#             input_pos_w=window_pos_edge_attrs,
#             input_neg_w=window_neg_edge_attrs,
#             target_pos_w=target_pos_edge_attrs,
#             target_neg_w=target_neg_edge_attrs,
#         )
#
#         return DTDGData(input=input, target=target, edge_index=edge_index, edge_weight=edge_weight,
#                      num_nodes=self.num_nodes)

import math
from typing import Any

import torch
import tsl
class DTDGData(tsl.data.Data):
    def __init__(self, input=None, target=None, edge_index=None, edge_weight=None, num_window_nodes=None, num_nodes=None):
        super().__init__(input=input, target=target, edge_index=edge_index, edge_weight=edge_weight)
        self.num_window_nodes = num_window_nodes
        self.num_nodes = num_nodes

    def __inc__(self, key, value, *args, **kwargs):
        if ('input' in key) and ('ei' in key):
            return self.num_window_nodes
        elif ('target' in key) and ('ei' in key):
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if (('input' in key) and ('ei' in key)) or (('target' in key) and ('ei' in key)):
            return 1
        return 0


class DTDGLinkDataset(torch.utils.data.Dataset):
    def __init__(self, pos_data, neg_data, window=11, horizon=1, stride=1):
        self.pos_xs, self.pos_edge_idxs, self.pos_edge_attrs = pos_data
        self.neg_xs, self.neg_edge_idxs, self.neg_edge_attrs = neg_data
        self.num_nodes = self.pos_xs[0].size()[0]
        self.window = window
        self.horizon = horizon
        self.stride = stride
        self.tot_samples = math.ceil((len(self.pos_xs) - (self.window + self.horizon)) / self.stride)

    def __len__(self):
        return self.tot_samples

    # def __getitem__(self, idx):
    #     start = idx * self.stride
    #     end = start + self.window + self.horizon
    #
    #     sample_pos_xs, sample_pos_edge_idxs, sample_pos_edge_attrs = self.pos_xs[start: end], self.pos_edge_idxs[
    #                                                                                           start: end], self.pos_edge_attrs[
    #                                                                                                        start: end]
    #     sample_neg_xs, sample_neg_edge_idxs, sample_neg_edge_attrs = self.neg_xs[start: end], self.neg_edge_idxs[
    #                                                                                           start: end], self.neg_edge_attrs[
    #                                                                                                        start: end]
    #
    #     window_pos_xs, window_pos_edge_idxs, window_pos_edge_attrs = sample_pos_xs[:self.window], sample_pos_edge_idxs[
    #                                                                                               :self.window], sample_pos_edge_attrs[
    #                                                                                                              :self.window]
    #     window_neg_xs, window_neg_edge_idxs, window_neg_edge_attrs = sample_neg_xs[:self.window], sample_neg_edge_idxs[
    #                                                                                               :self.window], sample_neg_edge_attrs[
    #                                                                                                              :self.window]
    #
    #     target_pos_xs, target_pos_edge_idxs, target_pos_edge_attrs = sample_pos_xs[-1], sample_pos_edge_idxs[-1], \
    #         sample_pos_edge_attrs[-1]
    #     target_neg_xs, target_neg_edge_idxs, target_neg_edge_attrs = sample_neg_xs[-1], sample_neg_edge_idxs[-1], \
    #         sample_neg_edge_attrs[-1]
    #     input = dict(
    #         pos_x=torch.cat(window_pos_xs, dim=0),
    #         neg_x=torch.cat(window_neg_xs, dim=0)
    #     )
    #     target = dict(
    #         pos_y=torch.cat(target_pos_xs, dim=0) if self.horizon > 1 else torch.cat([target_pos_xs], dim=0),
    #         neg_y=torch.cat(target_neg_xs, dim=0) if self.horizon > 1 else torch.cat([target_neg_xs], dim=0)
    #     )
    #     edge_index = dict(
    #         input_pos_ei=torch.cat(window_pos_edge_idxs, dim=1),
    #         input_neg_ei=torch.cat(window_neg_edge_idxs, dim=1),
    #         target_pos_ei=target_pos_edge_idxs,
    #         target_neg_ei=target_neg_edge_idxs,
    #     )
    #     edge_weight = dict(
    #         input_pos_w=torch.cat(window_pos_edge_attrs, dim=0),
    #         input_neg_w=torch.cat(window_neg_edge_attrs, dim=0),
    #         target_pos_w=target_pos_edge_attrs,
    #         target_neg_w=target_neg_edge_attrs,
    #     )
    #
    #     return CDTDGData(input=input, target=target, edge_index=edge_index, edge_weight=edge_weight,
    #                  num_nodes=self.num_nodes)

    # Offset and concatenate edge indices
    def __concatenate_edge_indices__(self, edge_indices, node_offsets):
        concatenated_edges = []
        offset = 0
        for edges, node_count in zip(edge_indices, node_offsets):
            concatenated_edges.append(edges + offset)
            offset += node_count
        return torch.cat(concatenated_edges, dim=1)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window + self.horizon

        sample_pos_xs, sample_pos_edge_idxs, sample_pos_edge_attrs = (
            self.pos_xs[start:end],
            self.pos_edge_idxs[start:end],
            self.pos_edge_attrs[start:end],
        )
        sample_neg_xs, sample_neg_edge_idxs, sample_neg_edge_attrs = (
            self.neg_xs[start:end],
            self.neg_edge_idxs[start:end],
            self.neg_edge_attrs[start:end],
        )

        window_pos_xs, window_pos_edge_idxs, window_pos_edge_attrs = (
            sample_pos_xs[:self.window],
            sample_pos_edge_idxs[:self.window],
            sample_pos_edge_attrs[:self.window],
        )
        window_neg_xs, window_neg_edge_idxs, window_neg_edge_attrs = (
            sample_neg_xs[:self.window],
            sample_neg_edge_idxs[:self.window],
            sample_neg_edge_attrs[:self.window],
        )

        target_pos_xs, target_pos_edge_idxs, target_pos_edge_attrs = (
            sample_pos_xs[-1],
            sample_pos_edge_idxs[-1],
            sample_pos_edge_attrs[-1],
        )
        target_neg_xs, target_neg_edge_idxs, target_neg_edge_attrs = (
            sample_neg_xs[-1],
            sample_neg_edge_idxs[-1],
            sample_neg_edge_attrs[-1],
        )

        # Concatenate node features
        pos_x = torch.cat(window_pos_xs, dim=0)
        neg_x = torch.cat(window_neg_xs, dim=0)

        pos_node_offsets = [x.size(0) for x in window_pos_xs]
        neg_node_offsets = [x.size(0) for x in window_neg_xs]

        concatenated_pos_edge_idxs = self.__concatenate_edge_indices__(window_pos_edge_idxs, pos_node_offsets)
        concatenated_neg_edge_idxs = self.__concatenate_edge_indices__(window_neg_edge_idxs, neg_node_offsets)

        input = dict(
            pos_x=pos_x,
            neg_x=neg_x
        )
        target = dict(
            pos_y=torch.cat(target_pos_xs, dim=0) if self.horizon > 1 else torch.cat([target_pos_xs], dim=0),
            neg_y=torch.cat(target_neg_xs, dim=0) if self.horizon > 1 else torch.cat([target_neg_xs], dim=0)
        )
        edge_index = dict(
            input_pos_ei=concatenated_pos_edge_idxs,
            input_neg_ei=concatenated_neg_edge_idxs,
            target_pos_ei=target_pos_edge_idxs,
            target_neg_ei=target_neg_edge_idxs,
        )
        edge_weight = dict(
            input_pos_w=torch.cat(window_pos_edge_attrs, dim=0),
            input_neg_w=torch.cat(window_neg_edge_attrs, dim=0),
            target_pos_w=target_pos_edge_attrs,
            target_neg_w=target_neg_edge_attrs,
        )

        return DTDGData(input=input, target=target, edge_index=edge_index, edge_weight=edge_weight,
                         num_window_nodes=pos_x.shape[0], num_nodes=self.num_nodes)
