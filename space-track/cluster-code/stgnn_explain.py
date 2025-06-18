import time
import argparse
import pandas as pd
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GDataLoader
import torch_geometric.nn as nng

from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from dynamic_graph_dataset import DTDGLinkDataset
from lightning_model import STGNN
from tqdm import tqdm
import torch_geometric as pyg
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import degree
from torch_geometric.explain import ExplainerConfig, ModelConfig, ThresholdConfig, Explainer, DummyExplainer, \
    GNNExplainer, PGExplainer
from torch_geometric.explain.config import ModelMode, MaskType
from torch_geometric.explain.metric import groundtruth_metrics, fidelity, characterization_score, fidelity_curve_auc, \
    unfaithfulness
from torchmetrics.functional.classification import binary_jaccard_index

from dataset_splitter_converter_sampler import sample_new, load_existing
from layers import TimeSpaceModel
from explainer_store import get_explainer
from link_pgexplainer import LinkPGExplainer


def compute_loss(pos_score, neg_score, device=None):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0], device=device), torch.zeros(neg_score.shape[0], device=device)]
    ).double()
    return F.binary_cross_entropy(scores, labels)


def reduce_concatenated_eis(concatenated_ei, num_nodes_per_graph, num_graphs):
    for graph_id in range(num_graphs):
        # Calculate the range of node IDs for the current graph
        min_node_id = graph_id * num_nodes_per_graph
        max_node_id = (graph_id + 1) * num_nodes_per_graph

        # Mask to identify edges belonging to the current graph
        mask = (
                (concatenated_ei[0] >= min_node_id) & (concatenated_ei[0] < max_node_id) &
                (concatenated_ei[1] >= min_node_id) & (concatenated_ei[1] < max_node_id)
        )

        # Extract edges for the current graph
        concatenated_ei[:, mask] = concatenated_ei[:, mask] - min_node_id  # Normalize to start at 0


def time_forward(input_tensor, encoder, batch_size, num_graphs_batch, window, num_nodes):
    # input_tensor: [batch_size * time_steps * num_nodes, features]
    # print(f'input_tensor: {input_tensor.shape}')

    reshaped_input = input_tensor.view(batch_size, int(input_tensor.shape[0] / batch_size),
                                       -1)  # [batch_size, time_steps * num_nodes, features]
    # print(f'reshaped_input: {reshaped_input.shape}')

    encoded_input = encoder.input_encoder(
        reshaped_input)  # Linear encoder: encoded_input = input_tensor * weights + bias [batch_size, time_steps * num_nodes, hidden_size]
    # print(f'encoded_input: {encoded_input.shape}')

    current_sequence_length = num_graphs_batch
    target_sequence_length = window
    padding_length = target_sequence_length - current_sequence_length
    # print(f'current_sequence_length: {current_sequence_length}')
    # print(f'target_sequence_length: {target_sequence_length}')
    # print(f'padding_length: {padding_length}')

    encoded_input_reshaped = encoded_input.view(batch_size, -1, num_nodes, encoded_input.shape[
        2])  # [batch_size, time_steps, num_nodes, hidden_size]
    # print(f'num_nodes: {num_nodes}')
    if batch_size != num_graphs_batch:
        padding_tensor = torch.zeros(batch_size, padding_length, num_nodes, encoded_input.shape[2]).to(
            encoded_input.device)
        encoded_input_reshaped = torch.cat([encoded_input_reshaped, padding_tensor], dim=1)

    # print(f'encoded_input_reshaped: {encoded_input_reshaped.shape}')  # [batch_size, time_steps, num_nodes, hidden_size]

    temporal_output = encoder.time_nn(
        encoded_input_reshaped) if encoder.time_nn is not None else encoded_input  # Temporal processing
    # print(f'temporal_output: {temporal_output.shape}')  # [batch_size, time_steps, num_nodes, hidden_size]

    reshaped_temporal_output_size = batch_size * window * num_nodes  # because tensor was padded with zeros it is necessary to use self.window instead of time_steps
    temporal_output_reshaped = temporal_output.reshape(reshaped_temporal_output_size, -1)
    if batch_size != num_graphs_batch:
        temporal_output_reshaped = temporal_output_reshaped[:input_tensor.shape[0], :]
    # print(f'temporal_output_reshaped: {temporal_output_reshaped.shape}')
    return temporal_output_reshaped


def evaluate_lf_explainer_on_data(explainer, data, edge_label_indices, metric_names, use_prob=False, max_links=10):
    eval_metrics = {metric_name: 0 for metric_name in metric_names}
    pos_fids, neg_fids, inf_times = [], [], []
    random_indices = [random.randint(0, edge_label_indices.size(1) - 1) for _ in range(max_links)]
    for idx in random_indices:
        start = time.time()
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            target=data.edge_label_prob[idx] if use_prob else data.edge_label[idx],
            edge_label_index=edge_label_indices[:, idx].view(-1, 1),
            num_graphs=data.num_graphs
        )
        end = time.time()
        inference_time = end - start
        inf_times.append(inference_time)
        pos_fidelity, neg_fidelity = fidelity(explainer, explanation)
        pos_fids.append(pos_fidelity)
        neg_fids.append(neg_fidelity)

    eval_metrics['fid+'] = np.mean(pos_fids)
    eval_metrics['fid-'] = np.mean(neg_fids)
    eval_metrics['inference_time'] = np.mean(inf_times)
    eval_metrics['characterization_score'] = characterization_score(eval_metrics['fid+'], eval_metrics['fid-'])
    return eval_metrics


def evaluate_lf_explainer(model, model_name, explainer_name, explainer_config, dataloader, metric_names, window,
                          num_nodes, max_iter=50, max_links=10, device='cuda'):
    start_time = time.time()

    print(f'{"-" * 2} Evaluating {explainer_name} explainer on SatCon...')
    exp_eval_metrics = {}
    model_config = ModelConfig(mode='binary_classification', task_level='edge', return_type='probs')
    threshold_config = ThresholdConfig(value=0, threshold_type='hard')
    num_iter = 0
    for batch in dataloader:
        batch_start_time = time.time()
        if num_iter >= max_iter:
            break
        with torch.no_grad():
            batch = batch.to(device)
            x_time = time_forward(batch.input.pos_x.float(), model.encoder, batch.batch_size, batch.num_graphs, window,
                                  num_nodes)
        xs = torch.chunk(x_time, window, dim=0)
        reduce_concatenated_eis(batch.edge_index['input_pos_ei'], num_nodes, window)
        data = Data(x=xs[-1], edge_index=batch.edge_index['input_pos_ei'], edge_weight=batch.edge_weight['input_pos_w'],
                    edge_label_index=batch.edge_index['target_pos_ei'],
                    edge_label=torch.ones(batch.edge_index['target_pos_ei'].size(1)), num_graphs=batch.num_graphs).to(device)

        pos_edges = data.edge_label_index
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
        degree_zero_nodes = torch.nonzero(deg == 0, as_tuple=True)[0]
        source_nodes = pos_edges[0]
        target_nodes = pos_edges[1]

        mask = ~torch.isin(source_nodes, degree_zero_nodes) & ~torch.isin(target_nodes, degree_zero_nodes)

        num_examples = 200
        edge_label_indices = pos_edges[:, mask][:, :num_examples]

        exp_eval_metrics[(explainer_name, model_name)] = {}

        use_prob = False
        explainer = get_explainer(explainer_name, explainer_config, model, model_config, threshold_config, dataset=data,
                                  edge_label_indices=edge_label_indices)
        if explainer_name == 'ciexplainer':
            use_prob = True
            edge_label_prob = torch.zeros_like(data.edge_label[:num_examples], dtype=torch.float64,
                                               device=data.x.device)
            for idx in range(edge_label_prob.size(0)):
                edge_label_prob[idx] = model(data.x, data.edge_index,
                                             edge_label_index=edge_label_indices[:, idx].view(-1, 1)).sigmoid()
            data.edge_label_prob = edge_label_prob
        res = evaluate_lf_explainer_on_data(explainer, data, edge_label_indices, metric_names, use_prob=use_prob,
                                            max_links=max_links)
        for metric_name, metric_value in res.items():
            exp_eval_metrics[(explainer_name, model_name)][('satcon', metric_name)] = metric_value

        batch_end_time = time.time()
        elapsed_time = (batch_end_time - batch_start_time) / 60
        print(f'{"-" * 3} Evaluation on Batch {num_iter} took {elapsed_time:.2f} minutes.')

        num_iter += 1
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f'{"-" * 2} Evaluation on SatCon took {elapsed_time:.2f} minutes.')
    return exp_eval_metrics


def evaluation_df(eval_data, explainer_names, model_names, dataset_names, metric_names):
    # Create MultiIndex for rows and columns
    row_index = pd.MultiIndex.from_product([explainer_names, model_names], names=['Explainer', 'Model'])
    col_index = pd.MultiIndex.from_product([dataset_names, metric_names], names=['Dataset', 'Metric'])

    # Convert the nested dictionary into a DataFrame
    dataframe = pd.DataFrame.from_dict(eval_data, orient='index')
    dataframe.index = row_index
    dataframe.columns = col_index

    return dataframe


def main():
    parser = argparse.ArgumentParser(
        description='Train a spatio-temporal graph neural network on the space-track dataset')
    parser.add_argument('--weight', type=int, default=-1,
                        help='The weight of the edge. Chose index from ["r_dist", "ct_dist", "it_dist", "dist"]')
    parser.add_argument('--data_folder_location', type=str, default='/data/f.caldas/ssk/',
                        help='The location of the data folder')
    parser.add_argument('--new_sample', type=bool, default=False, help='Whether to sample new negative edges')
    args = parser.parse_args()
    print(args)

    start_time = time.time()
    # ----------------------------------------- Importing and Preprocessing the Data -----------------------------------
    print("Importing and Preprocessing the Data...")
    if args.new_sample:
        nodes_df, train_pos_data, train_neg_data, val_pos_data, val_neg_data, test_pos_data, test_neg_data = sample_new(
            data_folder_location=args.data_folder_location, weight=args.weight)
    else:
        nodes_df, train_pos_data, train_neg_data, val_pos_data, val_neg_data, test_pos_data, test_neg_data = load_existing(
            data_folder_location=args.data_folder_location, weight=args.weight)

    # ---------------------------------------- Defining the dataset hyperparemeters ------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    window = 128  # should be < len(train_pos_data[0]) train_pos_data[0] == pos_xs
    horizon = 1
    shuffle = False
    batch_size = 1

    test_dataset = DTDGLinkDataset(test_pos_data, test_neg_data, window=window, horizon=horizon)
    test_loader = GDataLoader(test_dataset, shuffle=shuffle, batch_size=batch_size)
    batch = next(iter(test_loader))
    print(batch)

    # ------------------------------------------------ Load model ------------------------------------------------------

    num_nodes = len(list(nodes_df.index.unique()))  # 777
    input_size = len(nodes_df.columns) - 1  # 39
    hidden_size = 64  # args.hidden_size
    output_size = 128  # args.output_size
    time_layers = 1  # args.time_layers
    space_layers = 1  # args.space_layers

    time_modules = ['gru', 'lstm', 'multigru', 'multilstm', 'graphconvgru', 'graphconvlstm', 'dcrnn', 'evolvegcn',
                    'tranformer', 'cnn', 'gcnn', 'att', 'tcn', 'condtcn', 'stcn', None]
    time_module = time_modules[1]  # args.time_module
    space_modules = ['gcn', 'diffconv', 'gat', 'sage', 'gatconv', None]
    space_module = space_modules[0]  # args.space_module
    encoder = TimeSpaceModel(time_module=time_module, space_module=space_module, input_size=input_size,
                             output_size=output_size, hidden_size=hidden_size, time_layers=time_layers,
                             space_layers=space_layers, window=window,
                             horizon=1, num_nodes=num_nodes, batch_size=1).float()
    clf = nng.InnerProductDecoder()  # args.dec
    model_name = f'{time_module}-{space_module}'
    model = STGNN.load_from_checkpoint(
        f'{args.data_folder_location}/logs/stgnn-bce-{time_module}-1-{space_module}-1-h-64-o-128-w-128-e-100-b-32/version_1/checkpoints/cp.ckpt',
        encoder=encoder, clf=clf, loss_fn=compute_loss).float().to(device)
    model.eval()
    print(model)
    # --------------------------------------------------- Explain ------------------------------------------------------
    metric_names = ['fid+', 'fid-', 'characterization_score', 'inference_time']
    explainer_config = ExplainerConfig(explanation_type='phenomenon', node_mask_type='object', edge_mask_type='object')
    explainer = 'all'
    num_runs = 5
    save_eval_metrics = True
    metrics_save_path = 'eval_metrics/'
    max_iter = 50
    max_links = 100
    if explainer in ['random', 'all']:
        print('$' * 101)
        print('Evaluating random explainer...')
        rnd_start_time = time.time()
        for run in range(num_runs):
            print(f'Run {run}...')
            eval_metrics = evaluate_lf_explainer(model, model_name, 'random_explainer', explainer_config,
                                                 test_loader, metric_names, window, num_nodes, max_iter=max_iter,
                                                 max_links=max_links)
            eval_metrics_df = evaluation_df(eval_metrics, ['random_explainer'], [model_name],
                                            ['satcon'], metric_names)
            if save_eval_metrics:
                eval_metrics_df.to_csv(f'{metrics_save_path}random_explainer_lf_metrics_{run}.csv')
        rnd_end_time = time.time()
        rnd_elapsed = (rnd_end_time - rnd_start_time) / 60
        print(f'Random explainer took {rnd_elapsed:.2f} minutes')

    if explainer in ['gnnexplainer', 'all']:
        print('$' * 101)
        print('Evaluating GNNExplainer...')
        gnn_start_time = time.time()
        for run in range(num_runs):
            print(f'Run {run}...')
            eval_metrics = evaluate_lf_explainer(model, model_name, 'gnnexplainer', explainer_config,
                                                 test_loader, metric_names, window, num_nodes, max_iter=max_iter,
                                                 max_links=max_links)
            eval_metrics_df = evaluation_df(eval_metrics, ['gnnexplainer'], [model_name],
                                            ['satcon'], metric_names)
            if save_eval_metrics:
                eval_metrics_df.to_csv(f'{metrics_save_path}gnnexplainer_lf_metrics_{run}.csv')
        gnn_end_time = time.time()
        gnn_elapsed = (gnn_end_time - gnn_start_time) / 60
        print(f'GNNExplainer took {gnn_elapsed:.2f} minutes')

    if explainer in ['pgexplainer', 'all']:
        print('$' * 101)
        print('Evaluating PGExplainer...')
        pg_start_time = time.time()
        for run in range(num_runs):
            print(f'Run {run}...')
            eval_metrics = evaluate_lf_explainer(model, model_name, 'pgexplainer', explainer_config,
                                                 test_loader, metric_names, window, num_nodes, max_iter=max_iter,
                                                 max_links=max_links)
            eval_metrics_df = evaluation_df(eval_metrics, ['pgexplainer'], [model_name],
                                            ['satcon'], metric_names)
            if save_eval_metrics:
                eval_metrics_df.to_csv(f'{metrics_save_path}pgexplainer_lf_metrics_{run}.csv')
        pg_end_time = time.time()
        pg_elapsed = (pg_end_time - pg_start_time) / 60
        print(f'PGExplainer took {pg_elapsed:.2f} minutes')

    if explainer in ['ciexplainer']:
        print('$' * 101)
        print('Evaluating CIExplainer...')
        ci_start_time = time.time()
        for run in range(num_runs):
            print(f'Run {run}...')
            eval_metrics = evaluate_lf_explainer(model, model_name, 'ciexplainer', explainer_config,
                                                 test_loader, metric_names, window, num_nodes, max_iter=max_iter,
                                                 max_links=max_links)
            eval_metrics_df = evaluation_df(eval_metrics, ['ciexplainer'], [model_name],
                                            ['satcon'], metric_names)
            if save_eval_metrics:
                eval_metrics_df.to_csv(f'{metrics_save_path}ciexplainer_lf_metrics_{run}.csv')
        ci_end_time = time.time()
        ci_elapsed = (ci_end_time - ci_start_time) / 60
        print(f'CIExplainer took {ci_elapsed:.2f} minutes')


if __name__ == '__main__':
    main()
