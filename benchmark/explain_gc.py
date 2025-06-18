import time
import numpy as np
import torch

from torch_geometric.explain import ModelConfig
from torch_geometric.explain.metric import groundtruth_metrics, fidelity, characterization_score, unfaithfulness
from torchmetrics.functional.classification import binary_jaccard_index

from utils import setup_models, custom_iou, custom_precision, custom_recall, custom_fidelity
from model_store import get_gnn
from explainer_store import get_explainer


def evaluate_gc_explainer_on_data(explainer, data_list, metric_names, use_prob=False, gt_metrics=None, threshold=0.5):
    eval_metrics = {metric_name: 0 for metric_name in metric_names}
    if gt_metrics is None:
        gt_metrics = ['accuracy', 'auroc']
    accs, precisions, recalls, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], [], []
    for data in data_list:
        start = time.time()
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=explainer.model(data.x, data.edge_index).sigmoid() if use_prob else data.y.unsqueeze(0)
        )
        end = time.time()
        inference_time = end - start
        inf_times.append(inference_time)
        data_nodes = data.edge_index[:, data.edge_mask.bool()].view(-1).unique()
        data_node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device, requires_grad=False)
        data_node_mask[data_nodes] = 1

        if 'node_mask' in explanation:
            node_mask = explanation.node_mask
        else:
            nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
            node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
            node_mask[nodes] = 1

        explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
        gt_nodes = (torch.nonzero(data_node_mask.view(-1) == 1).squeeze()).detach().cpu()
        exp_nodes = (node_mask.view(-1).sort(descending=True)[1][:gt_nodes.size(0)]).detach().cpu()

        iou = custom_iou(gt_nodes, exp_nodes)
        precision = custom_precision(gt_nodes, exp_nodes)
        recall = custom_recall(gt_nodes, exp_nodes)
        gt, pred = data_node_mask.view(-1), node_mask.view(-1)
        acc = groundtruth_metrics(pred, gt, metrics=['accuracy'], threshold=threshold)
        # auroc = groundtruth_metrics(pred, gt, metrics=['auroc'])
        # iou = binary_jaccard_index(pred, gt, threshold=threshold).cpu()
        pos_fidelity, neg_fidelity = custom_fidelity(explainer, explanation, node_mask, max_nodes=gt_nodes.size(0), full_graph=True)
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


def evaluate_gc_explainer(model_path, explainer_name, explainer_config, gc_datasets, metric_names, std=None):
    start_time = time.time()

    print(f'{"-" * 2} Evaluating {explainer_name} explainer on graph classification datasets...')
    exp_eval_metrics = {}
    model_config = ModelConfig(mode='binary_classification', task_level='graph', return_type='raw')
    for dataset_name, test_data_list, num_classes in gc_datasets:
        gc_dataset_start_time = time.time()
        print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')

        gc_models = get_gnn(model_path, 'gc', 'all', dataset_name, std='none' if std is None else std)
        gc_models = setup_models(gc_models, test_data_list[0].x.device)

        for model_name, model in gc_models:
            gc_model_start_time = time.time()
            model_name = f'{model_name}-{std}'
            print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
            if (explainer_name, model_name) not in exp_eval_metrics:
                exp_eval_metrics[(explainer_name, model_name)] = {}

            use_prob = explainer_name == 'ciexplainer'

            cat_feat_indices = None
            if dataset_name == 'mutag':
                cat_feat_indices = [0]
            explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=test_data_list,
                                      cat_feat_indices=cat_feat_indices)
            threshold = 0.5
            if explainer_name == 'ciexplainer':
                threshold = 0.0
            res = evaluate_gc_explainer_on_data(explainer, test_data_list, metric_names, use_prob, gt_metrics=None,
                                                threshold=threshold)
            for metric_name, metric_value in res.items():
                exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value

            gc_model_end_time = time.time()
            gc_mode_elapsed_time = (gc_model_end_time - gc_model_start_time) / 60
            print(f'{"-" * 7} Evaluation on {model_name} model took {gc_mode_elapsed_time:.2f} minutes.')

        gc_dataset_end_time = time.time()
        gc_dataset_elapsed_time = (gc_dataset_end_time - gc_dataset_start_time) / 60
        print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {gc_dataset_elapsed_time:.2f} minutes.')

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f'{"-" * 3} Evaluation on graph classification took {elapsed_time:.2f} minutes.')
    return exp_eval_metrics

# def evaluate_gc_explainer_on_data(explainer, data_list, metric_names, use_prob=False, gt_metrics=None, threshold=0.5):
#     eval_metrics = {metric_name: 0 for metric_name in metric_names}
#     if gt_metrics is None:
#         gt_metrics = ['accuracy', 'auroc']
#     accs, aurocs, ious, pos_fids, neg_fids, us, inf_times = [], [], [], [], [], [], []
#     for data in data_list:
#         start = time.time()
#         explanation = explainer(
#             x=data.x,
#             edge_index=data.edge_index,
#             target=explainer.model(data.x, data.edge_index).sigmoid() if use_prob else data.y.unsqueeze(0)
#         )
#         end = time.time()
#         inference_time = end - start
#         inf_times.append(inference_time)
#         data_nodes = data.edge_index[:, data.edge_mask.bool()].view(-1).unique()
#         data_node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device, requires_grad=False)
#         data_node_mask[data_nodes] = 1
#
#         if 'node_mask' in explanation:
#             node_mask = explanation.node_mask
#         else:
#             nodes = explanation.edge_index[:, explanation.edge_mask.bool()].view(-1).unique()
#             node_mask = torch.zeros(data.x.size(0), 1, device=data.x.device)
#             node_mask[nodes] = 1
#         explanation.edge_mask = explanation.edge_mask.float() if explanation.edge_mask.dtype == torch.bool else explanation.edge_mask
#         gt, pred = data_node_mask.view(-1), node_mask.view(-1)
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
# def evaluate_gc_explainer(model_path, explainer_name, explainer_config, gc_datasets, metric_names):
#     start_time = time.time()
#
#     print(f'{"-" * 2} Evaluating {explainer_name} explainer on graph classification datasets...')
#     exp_eval_metrics = {}
#     model_config = ModelConfig(mode='binary_classification', task_level='graph', return_type='raw')
#     for dataset_name, test_data_list, num_classes in gc_datasets:
#         gc_dataset_start_time = time.time()
#
#         print(f'{"-" * 3} Evaluating {explainer_name} explainer on {dataset_name} dataset...')
#         gc_models = get_gnn(model_path, 'gc', 'all', dataset_name)
#         gc_models = setup_models(gc_models, test_data_list[0].x.device)
#
#         for model_name, model in gc_models:
#             gc_model_start_time = time.time()
#
#             print(f'{"-" * 5} Evaluating {explainer_name} explainer on {model_name} model...')
#             if (explainer_name, model_name) not in exp_eval_metrics:
#                 exp_eval_metrics[(explainer_name, model_name)] = {}
#
#             use_prob = explainer_name == 'ciexplainer'
#
#             cat_feat_indices = None
#             if dataset_name == 'mutag':
#                 cat_feat_indices = [0]
#             explainer = get_explainer(explainer_name, explainer_config, model, model_config, dataset=test_data_list,
#                                       dataset_name=dataset_name, cat_feat_indices=cat_feat_indices)
#             threshold = 0.5
#             if explainer_name == 'ciexplainer':
#                 threshold = 0.0
#             res = evaluate_gc_explainer_on_data(explainer, test_data_list, metric_names, use_prob, gt_metrics=None,
#                                                 threshold=threshold)
#             for metric_name, metric_value in res.items():
#                 exp_eval_metrics[(explainer_name, model_name)][(dataset_name, metric_name)] = metric_value
#
#             gc_model_end_time = time.time()
#             gc_mode_elapsed_time = (gc_model_end_time - gc_model_start_time) / 60
#             print(f'{"-" * 7} Evaluation on {model_name} model took {gc_mode_elapsed_time:.2f} minutes.')
#
#         gc_dataset_end_time = time.time()
#         gc_dataset_elapsed_time = (gc_dataset_end_time - gc_dataset_start_time) / 60
#         print(f'{"-" * 6} Evaluation on {dataset_name} dataset took {gc_dataset_elapsed_time:.2f} minutes.')
#
#     end_time = time.time()
#     elapsed_time = (end_time - start_time) / 60
#     print(f'{"-" * 3} Evaluation on graph classification took {elapsed_time:.2f} minutes.')
#     return exp_eval_metrics
