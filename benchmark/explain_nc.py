import time

import numpy as np
import torch

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import ModelConfig
from torch_geometric.explain.config import ModelMode
from torch_geometric.explain.metric import groundtruth_metrics, fidelity, characterization_score, unfaithfulness
from torchmetrics.functional.classification import binary_jaccard_index

from model_store import get_gnn
from explainer_store import get_explainer
from utils import setup_models, custom_iou, custom_precision, custom_recall, custom_fidelity, get_motif_nodes


def evaluate_nc_explainer_on_data(explainer, data, node_indices, metric_names, test_nodes_start, test_nodes_end,
                                  num_motif_nodes, use_prob=False, gt_metrics=None,
                                  threshold=0.5):
    eval_metrics = {metric_name: 0 for metric_name in metric_names}
    if gt_metrics is None:
        gt_metrics = ['accuracy', 'auroc']
    accs, precisions, recalls, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], [], []
    for idx in node_indices:
        start = time.time()
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=data.y_p if use_prob else data.y,
            index=idx.item(),
        )
        end = time.time()
        inference_time = end - start
        inf_times.append(inference_time)
        sub_nodes, _, _, hard_edge_mask = k_hop_subgraph(explanation.index.item(),
                                                         num_hops=explainer.model.model.num_layers,
                                                         edge_index=explanation.edge_index)
        explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
        if 'node_mask' in explanation:
            node_mask = explanation.node_mask
        else:
            nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
            node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
            node_mask[nodes] = 1
        exp_nodes = (node_mask.view(-1).sort(descending=True)[1][:num_motif_nodes]).detach().cpu()
        gt_nodes = get_motif_nodes(test_nodes_start, test_nodes_end, num_motif_nodes, idx.item()).detach().cpu()
        iou = custom_iou(gt_nodes, exp_nodes)
        precision = custom_precision(gt_nodes, exp_nodes)
        recall = custom_recall(gt_nodes, exp_nodes)
        gt, pred = data.node_mask[sub_nodes], node_mask[sub_nodes].view(-1)
        acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=threshold)

        # iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
        pos_fidelity, neg_fidelity = custom_fidelity(explainer, explanation, node_mask, max_nodes=num_motif_nodes)
        u = unfaithfulness(explainer, explanation)
        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        ious.append(iou)
        pos_fids.append(pos_fidelity)
        neg_fids.append(neg_fidelity)
        us.append(u)

    eval_metrics['accuracy'] = np.mean(accs)
    eval_metrics['precision'] = np.mean(precisions)
    eval_metrics['recall'] = np.mean(recalls)
    eval_metrics['iou'] = np.mean(ious)
    eval_metrics['fid+'] = np.mean(pos_fids)
    eval_metrics['fid-'] = np.mean(neg_fids)
    eval_metrics['unfaithfulness'] = np.mean(us)
    eval_metrics['inference_time'] = np.mean(inf_times)
    eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
    return eval_metrics


def evaluate_nc_explainer(model_path, explainer_name, explainer_config, nc_datasets, metric_names, std=None):
    start_time = time.time()

    print(f'{"-" * 2} Evaluating {explainer_name} explainer on node classification datasets...')
    exp_eval_metrics = {}
    model_config = ModelConfig(mode='multiclass_classification', task_level='node', return_type='raw')
    for dataset_name, dataset in nc_datasets:
        nc_dataset_start_time = time.time()
        motif_nodes = (dataset.y > 0).nonzero().view(-1)
        start = motif_nodes[0].item()
        end = motif_nodes[-1].item()
        step = 5 if 'ba' in dataset_name else 9
        print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
        nc_models = get_gnn(model_path, 'nc', 'all', dataset_name, std='all' if std is None else std)
        nc_models = setup_models(nc_models, dataset.x.device)
        motif_nodes_mask = (dataset.y > 0) & dataset.test_mask
        node_indices = motif_nodes_mask.nonzero().view(-1)
        if 'tree' in dataset_name:
            model_config.mode = ModelMode.binary_classification

        for model_name, model in nc_models:
            nc_model_start_time = time.time()
            model_name = f'{model_name}-{std}'
            print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
            if (explainer_name, model_name) not in exp_eval_metrics:
                exp_eval_metrics[(explainer_name, model_name)] = {}

            use_prob = False
            explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=dataset)
            threshold = 0.5
            if explainer_name == 'ciexplainer':
                use_prob = True
                threshold = 0.0
                y_p = torch.zeros_like(dataset.y, dtype=torch.float64, device=dataset.y.device)
                for idx in node_indices:
                    if explainer.model.model.out_channels == 1:
                        y_p[idx] = torch.sigmoid(model(dataset.x, dataset.edge_index)[idx])
                    else:
                        y_p[idx] = torch.softmax(model(dataset.x, dataset.edge_index)[idx], dim=-1).max()
                dataset.y_p = y_p
            res = evaluate_nc_explainer_on_data(explainer, dataset, node_indices, metric_names, start, end, step,
                                                use_prob=use_prob,
                                                gt_metrics=None, threshold=threshold)
            for metric_name, metric_value in res.items():
                exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value

            nc_model_end_time = time.time()
            nc_mode_elapsed_time = (nc_model_end_time - nc_model_start_time) / 60
            print(f'{"-" * 7} Evaluation on {model_name} model took {nc_mode_elapsed_time:.2f} minutes.')

        nc_dataset_end_time = time.time()
        nc_dataset_elapsed_time = (nc_dataset_end_time - nc_dataset_start_time) / 60
        print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {nc_dataset_elapsed_time:.2f} minutes.')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f'{"-" * 3} Evaluation on node classification took {elapsed_time:.2f} minutes.')
    return exp_eval_metrics

# def evaluate_nc_explainer_on_data(explainer, data, node_indices, metric_names, use_prob=False, gt_metrics=None,
#                                   threshold=0.5):
#     eval_metrics = {metric_name: 0 for metric_name in metric_names}
#     if gt_metrics is None:
#         gt_metrics = ['accuracy', 'auroc']
#     accs, aurocs, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], []
#     for idx in node_indices:
#         start = time.time()
#         explanation = explainer(
#             x=data.x,
#             edge_index=data.edge_index,
#             target=data.y_p if use_prob else data.y,
#             index=idx.item(),
#         )
#         end = time.time()
#         inference_time = end - start
#         inf_times.append(inference_time)
#         sub_nodes, _, _, hard_edge_mask = k_hop_subgraph(explanation.index.item(),
#                                                          num_hops=explainer.model.model.num_layers,
#                                                          edge_index=explanation.edge_index)
#         explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
#         if 'node_mask' in explanation:
#             node_mask = explanation.node_mask
#         else:
#             nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
#             node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
#             node_mask[nodes] = 1
#
#         gt, pred = data.node_mask[sub_nodes], node_mask[sub_nodes].view(-1)
#         acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=threshold)
#         auroc = groundtruth_metrics(pred, gt, metrics=['auroc'])
#         iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
#         pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
#         u = unfaithfulness(explainer, explanation)
#         accs.append(acc)
#         aurocs.append(auroc)
#         ious.append(iou)
#         pos_fids.append(pos_fidelity)
#         neg_fids.append(neg_fidelity)
#         us.append(u)
#
#     eval_metrics['accuracy'] = np.mean(accs)
#     eval_metrics['auroc'] = np.mean(aurocs)
#     eval_metrics['iou'] = np.mean(ious)
#     eval_metrics['fid+'] = np.mean(pos_fids)
#     eval_metrics['fid-'] = np.mean(neg_fids)
#     eval_metrics['unfaithfulness'] = np.mean(us)
#     eval_metrics['inference_time'] = np.mean(inf_times)
#     eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
#     return eval_metrics
#
#
# def evaluate_nc_explainer(model_path, explainer_name, explainer_config, nc_datasets, metric_names):
#     start_time = time.time()
#
#     print(f'{"-" * 2} Evaluating {explainer_name} explainer on node classification datasets...')
#     exp_eval_metrics = {}
#     model_config = ModelConfig(mode='multiclass_classification', task_level='node', return_type='raw')
#     for dataset_name, dataset in nc_datasets:
#         nc_dataset_start_time = time.time()
#
#         print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
#         nc_models = get_gnn(model_path, 'nc', 'all', dataset_name)
#         nc_models = setup_models(nc_models, dataset.x.device)
#         motif_nodes_mask = (dataset.y > 0) & dataset.test_mask
#         node_indices = motif_nodes_mask.nonzero().view(-1)
#         if 'tree' in dataset_name:
#             model_config.mode = ModelMode.binary_classification
#
#         for model_name, model in nc_models:
#             nc_model_start_time = time.time()
#
#             print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
#             if (explainer_name, model_name) not in exp_eval_metrics:
#                 exp_eval_metrics[(explainer_name, model_name)] = {}
#
#             use_prob = False
#             explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=dataset,
#                                       dataset_name=dataset_name)
#             threshold = 0.5
#             if explainer_name == 'ciexplainer':
#                 use_prob = True
#                 threshold = 0.0
#                 y_p = torch.zeros_like(dataset.y, dtype=torch.float64, device=dataset.y.device)
#                 for idx in node_indices:
#                     if explainer.model.model.out_channels == 1:
#                         y_p[idx] = torch.sigmoid(model(dataset.x, dataset.edge_index)[idx])
#                     else:
#                         y_p[idx] = torch.softmax(model(dataset.x, dataset.edge_index)[idx], dim=-1).max()
#                 dataset.y_p = y_p
#             res = evaluate_nc_explainer_on_data(explainer, dataset, node_indices, metric_names, use_prob=use_prob,
#                                                 gt_metrics=None, threshold=threshold)
#             for metric_name, metric_value in res.items():
#                 exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value
#
#             nc_model_end_time = time.time()
#             nc_mode_elapsed_time = (nc_model_end_time - nc_model_start_time) / 60
#             print(f'{"-" * 7} Evaluation on {model_name} model took {nc_mode_elapsed_time:.2f} minutes.')
#
#         nc_dataset_end_time = time.time()
#         nc_dataset_elapsed_time = (nc_dataset_end_time - nc_dataset_start_time) / 60
#         print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {nc_dataset_elapsed_time:.2f} minutes.')
#
#     end_time = time.time()
#     elapsed_time = (end_time - start_time) / 60
#     print(f'{"-" * 3} Evaluation on node classification took {elapsed_time:.2f} minutes.')
#     return exp_eval_metrics
