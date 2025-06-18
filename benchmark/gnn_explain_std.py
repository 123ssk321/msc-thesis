import random
import time
import argparse

import numpy as np
import pandas as pd

import torch
import torch_geometric as pyg
from torch_geometric.explain import ExplainerConfig
from torch_geometric.utils import degree

import data_store
import model_store
import explainer_store
from ciexplainer import CIExplainer
from explain_nc import evaluate_nc_explainer
from explain_gc import evaluate_gc_explainer
from explain_lp import evaluate_lp_explainer
from utils import setup_models


def evaluation_df(eval_data, explainer_names, model_names, dataset_names, metric_names):
    # Flatten the nested dictionary into a list of rows
    rows = []
    for (explainer, model), dataset_metrics in eval_data.items():
        for dataset in dataset_names:
            row = {
                'explainer': explainer,
                'dataset': dataset
            }

            # Split the model name into 'model' (before the hyphen) and 'std' (after the hyphen)
            model_parts = model.split('-')
            if len(model_parts) == 2:
                row['model'] = model_parts[0]  # e.g., 'gcn'
                row['std'] = model_parts[1]  # e.g., '00' (store as 'std')
            else:
                row['model'] = model  # If no hyphen, keep the entire model as 'model'
                row['std'] = None  # No model ID, set 'std' as None

            # Add metric values to the row
            for metric in metric_names:
                row[metric] = dataset_metrics.get((dataset, metric), None)

            rows.append(row)

    # Create DataFrame from the list of rows
    dataframe = pd.DataFrame(rows)

    # Reorder columns: model and std columns should come before accuracy
    column_order = ['explainer', 'dataset', 'model', 'std'] + metric_names
    dataframe = dataframe[column_order]

    return dataframe


def main(task, model, dataset, explainer, num_runs, std_idx):
    print(f'NUMBER OF EXPERIMENTS: {num_runs}')
    dataset_path = 'datasets/'
    model_save_path = 'models/'
    metrics_save_path = 'eval_metrics/'
    save_eval_metrics = True
    model_names = model_store.model_names
    explainer_names = explainer_store.explainer_names

    gcn = pyg.nn.GCN(in_channels=-1, hidden_channels=1, num_layers=1)
    grapsage = pyg.nn.GraphSAGE(in_channels=-1, hidden_channels=1, num_layers=1)
    gat = pyg.nn.GAT(in_channels=-1, hidden_channels=1, num_layers=1)
    gin = pyg.nn.GIN(in_channels=-1, hidden_channels=1, num_layers=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explainer_config = ExplainerConfig(explanation_type='phenomenon', node_mask_type='object', edge_mask_type='object')
    metric_names = ['accuracy', 'precision', 'recall', 'iou', 'fid+', 'fid-', 'unfaithfulness', 'characterization_score', 'inference_time']
    stds = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    std_str = stds[std_idx]
    print(f'STANDARD DEVIATION: {std_str}')
    if task in ['nc', 'all']:
        nc_datasets = data_store.get_nc_dataset(dataset_path, 'all', std_str=std_str, new=False)
        for idx, (dataset_name, data) in enumerate(nc_datasets):
            nc_datasets[idx] = (dataset_name, data.to(device))

    if task in ['gc', 'all']:
        gc_datasets = data_store.get_gc_dataset(dataset_path, 'all', std_str=std_str, explain=True)
        for i, (dataset_name, data_list, num_classes) in enumerate(gc_datasets):
            for j, data in enumerate(data_list):
                if 'ba' in dataset_name:
                    data.edge_mask = torch.logical_and(data.edge_index[0] >= 20, data.edge_index[1] >= 20)
                data_list[i] = data.to(device)
            #random.shuffle(data_list)
            train_size = int(0.8* len(data_list))
            val_size = int(0.1 * len(data_list))
            test_data_list = data_list[train_size + val_size:]
            gc_datasets[i] = (dataset_name, test_data_list, num_classes)

    if task in ['lp', 'all']:
        lp_datasets = data_store.get_lp_dataset(dataset_path, 'all', std=std_str)
        for idx, (dataset_name, train_data, val_data, test_data) in enumerate(lp_datasets):
            lp_datasets[idx] = (dataset_name, train_data.to(device), val_data.to(device), test_data.to(device))

    start = time.time()
    if explainer in ['random', 'all']:
        print('$' * 101)
        print('Evaluating random explainer...')
        rnd_start_time = time.time()
        if task in ['nc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_nc_explainer(model_save_path, 'random_explainer', explainer_config, nc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['random_explainer'], model_names, data_store.nc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}random_explainer_nc_metrics_std_{std_str}.csv')
        if task in ['gc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_gc_explainer(model_save_path, 'random_explainer', explainer_config, gc_datasets,
                                                  metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['random_explainer'], model_names, data_store.gc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}random_explainer_gc_metrics_std_{std_str}.csv')

        if task in ['lp', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_lp_explainer(model_save_path, 'random_explainer', explainer_config, lp_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['random_explainer'], model_names, data_store.lp_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}random_explainer_lp_metrics_std_{std_str}.csv')

        rnd_end_time = time.time()
        rnd_elapsed = (rnd_end_time - rnd_start_time) / 60
        print(f'Random explainer took {rnd_elapsed:.2f} minutes')

    if explainer in ['gnnexplainer', 'all']:
        print('$' * 101)
        print('Evaluating GNNExplainer...')
        gnn_start_time = time.time()
        if task in ['nc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_nc_explainer(model_save_path, 'gnnexplainer', explainer_config, nc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['gnnexplainer'], model_names, data_store.nc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}gnnexplainer_nc_metrics_std_{std_str}.csv')
        if task in ['gc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_gc_explainer(model_save_path, 'gnnexplainer', explainer_config, gc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['gnnexplainer'], model_names, data_store.gc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}gnnexplainer_gc_metrics_std_{std_str}.csv')
        if task in ['lp', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_lp_explainer(model_save_path, 'gnnexplainer', explainer_config, lp_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['gnnexplainer'], model_names, data_store.lp_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}gnnexplainer_lp_metrics_std_{std_str}.csv')

        gnn_end_time = time.time()
        gnn_elapsed = (gnn_end_time - gnn_start_time) / 60
        print(f'GNNExplainer took {gnn_elapsed:.2f} minutes')

    if explainer in ['pgexplainer', 'all']:
        print('$' * 101)
        print('Evaluating PGExplainer...')
        pg_start_time = time.time()
        if task in ['nc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_nc_explainer(model_save_path, 'pgexplainer', explainer_config, nc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['pgexplainer'], model_names, data_store.nc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}pgexplainer_nc_metrics_std_{std_str}.csv')
        if task in ['gc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_gc_explainer(model_save_path, 'pgexplainer', explainer_config, gc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['pgexplainer'], model_names, data_store.gc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}pgexplainer_gc_metrics_std_{std_str}.csv')
        if task in ['lp', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_lp_explainer(model_save_path, 'pgexplainer', explainer_config, lp_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['pgexplainer'], model_names, data_store.lp_datasets,
                metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}pgexplainer_lp_metrics_std_{std_str}.csv')

        pg_end_time = time.time()
        pg_elapsed = (pg_end_time - pg_start_time) / 60
        print(f'PGExplainer took {pg_elapsed:.2f} minutes')

    if explainer in ['subgraphx']:
        print('$' * 101)
        print('Evaluating SubgraphX...')
        sgx_start_time = time.time()
        if task in ['nc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_nc_explainer(model_save_path, 'subgraphx', explainer_config, nc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['subgraphx'], model_names, data_store.nc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}subgraphX_nc_metrics_std_{std_str}.csv')

        if task in ['gc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_gc_explainer(model_save_path, 'subgraphx', explainer_config, gc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['subgraphx'], model_names, data_store.gc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}subgraphx_gc_metrics_std_{std_str}.csv')
                
        if task in ['lp', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_lp_explainer(model_save_path, 'subgraphx', explainer_config, lp_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['subgraphx'], model_names, data_store.lp_datasets,
                                        metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}subgraphx_lp_metrics_std_{std_str}.csv')
                torch.cuda.empty_cache()

        sgx_end_time = time.time()
        sgx_elapsed = (sgx_end_time - sgx_start_time) / 60
        print(f'SubgraphX took {sgx_elapsed:.2f} minutes')

    if explainer in ['ciexplainer', 'all']:
        print('$' * 101)
        print('Evaluating CIExplainer...')
        ci_start_time = time.time()
        if task in ['nc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_nc_explainer(model_save_path, 'ciexplainer', explainer_config, nc_datasets,metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['ciexplainer'], model_names, data_store.nc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)

            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}ciexplainer_nc_metrics_std_{std_str}.csv')
        if task in ['gc', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_gc_explainer(model_save_path, 'ciexplainer', explainer_config, gc_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['ciexplainer'], model_names, data_store.gc_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}ciexplainer_gc_metrics_std_{std_str}.csv')
        if task in ['lp', 'all']:
            eval_dfs = []
            for run in range(num_runs):
                print(f'Run {run}...')
                eval_data = evaluate_lp_explainer(model_save_path, 'ciexplainer', explainer_config, lp_datasets, metric_names, std=std_str)
                eval_df = evaluation_df(eval_data, ['ciexplainer'], model_names, data_store.lp_datasets, metric_names)
                eval_df['run'] = run
                eval_dfs.append(eval_df)
            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if save_eval_metrics:
                eval_df.to_csv(f'{metrics_save_path}ciexplainer_lp_metrics_std_{std_str}.csv')
            ci_end_time = time.time()
            ci_elapsed = (ci_end_time - ci_start_time) / 60
            print(f'CIExplainer took {ci_elapsed:.2f} minutes')

    end = time.time()
    elapsed = (end - start) / 60
    print(f'{"-" * 101}')
    print(f'Evaluation took {elapsed:.2f} minutes')
    print(f'NUMBER OF EXPERIMENTS: {num_runs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Explain GNN models.')
    parser.add_argument('--task', type=str, default='all',
                        help='Task to perform: node classification (nc), graph classification (gc), link prediction (lp), or all.')
    parser.add_argument('--model', type=str, default='all', help='Model to use: gcn, gin, gat, graphsage, or all.')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Name of the dataset to use: ba_shapes(nc), ba_community(nc), tree_cycles(nc), tree_grid(nc), ba_2motif(gc), mutag(gc), or all.')
    parser.add_argument('--explainer', type=str, default='all',
                        help='Explainer to use: random, gnnexplainer, pgexplainer, subgraphX, ciexplainer, or all.')
    parser.add_argument('--num_runs', type=int, default=10, help='Number of runs to perform.')
    parser.add_argument('--std_idx', type=int, default=0, help='Index of the standard deviation to use (0-10).')
    args = parser.parse_args()
    main(args.task, args.model, args.dataset, args.explainer, args.num_runs, args.std_idx)
