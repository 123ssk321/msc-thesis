import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_store import get_nc_dataset, get_gc_dataset, get_lp_dataset
from graph_classification import graph_classification
from link_prediction import link_prediction
from node_classification import node_classification


def plot_loss(ax, train_loss, val_loss):
    color = 'tab:red'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color=color)
    ax.plot(train_loss, color=color, label='Train Loss')
    ax.plot(val_loss, color='tab:orange', linestyle='--', label='Val Loss')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


def plot_accuracy(ax, train_acc, val_acc):
    color = 'tab:blue'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy', color=color)
    ax.plot(train_acc, color=color, label='Train Accuracy')
    ax.plot(val_acc, color='tab:cyan', linestyle='--', label='Val Accuracy')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


def plot_f1(ax, train_f1, val_f1):
    color = 'tab:green'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('F1 Score', color=color)
    ax.plot(train_f1, color=color, label='Train F1 Score')
    ax.plot(val_f1, color='tab:olive', linestyle='--', label='Val F1 Score')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


def plot_precision(ax, train_precision, val_precision):
    color = 'tab:purple'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Precision', color=color)
    ax.plot(train_precision, color=color, label='Train Precision')
    ax.plot(val_precision, color='tab:pink', linestyle='--', label='Val Precision')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


def plot_recall(ax, train_recall, val_recall):
    color = 'tab:orange'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Recall', color=color)
    ax.plot(train_recall, color=color, label='Train Recall')
    ax.plot(val_recall, color='tab:brown', linestyle='--', label='Val Recall')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


def save_figure(fig, save_path, save):
    if save:
        fig.savefig(save_path, format='pdf')


def plot_metrics(metrics_df, save_path, save=False):
    # Loss
    train_loss = metrics_df['train_loss'].dropna().reset_index(drop=True)
    val_loss = metrics_df['val_loss'].dropna().reset_index(drop=True)
    fig_loss, ax1 = plt.subplots()
    plot_loss(ax1, train_loss, val_loss)
    fig_loss.tight_layout()
    loss_figure_savepath = f'{save_path}-loss.pdf'
    save_figure(fig_loss, loss_figure_savepath, save)

    # Accuracy
    train_acc = metrics_df['train_accuracy'].dropna().reset_index(drop=True)
    val_acc = metrics_df['val_accuracy'].dropna().reset_index(drop=True)
    fig_acc, ax2 = plt.subplots()
    plot_accuracy(ax2, train_acc, val_acc)
    fig_acc.tight_layout()
    acc_figure_savepath = f'{save_path}-accuracy.pdf'
    save_figure(fig_acc, acc_figure_savepath, save)

    # F1 Score
    train_f1 = metrics_df['train_f1'].dropna().reset_index(drop=True)
    val_f1 = metrics_df['val_f1'].dropna().reset_index(drop=True)
    fig_f1, ax3 = plt.subplots()
    plot_f1(ax3, train_f1, val_f1)
    fig_f1.tight_layout()
    f1_figure_savepath = f'{save_path}-f1.pdf'
    save_figure(fig_f1, f1_figure_savepath, save)

    # Precision
    train_precision = metrics_df['train_precision'].dropna().reset_index(drop=True)
    val_precision = metrics_df['val_precision'].dropna().reset_index(drop=True)
    fig_precision, ax4 = plt.subplots()
    plot_precision(ax4, train_precision, val_precision)
    fig_precision.tight_layout()
    precision_figure_savepath = f'{save_path}-precision.pdf'
    save_figure(fig_precision, precision_figure_savepath, save)

    # Recall
    train_recall = metrics_df['train_recall'].dropna().reset_index(drop=True)
    val_recall = metrics_df['val_recall'].dropna().reset_index(drop=True)
    fig_recall, ax5 = plt.subplots()
    plot_recall(ax5, train_recall, val_recall)
    fig_recall.tight_layout()
    recall_figure_savepath = f'{save_path}-recall.pdf'
    save_figure(fig_recall, recall_figure_savepath, save)


def plot(loggers, figure_save_path, save=False):
    for dataset_name, logger in loggers:
        print(f'Plotting metrics for {dataset_name} dataset')
        metrics_df = pd.read_csv(logger.log_dir + '/metrics.csv')
        plot_metrics(metrics_df, f'{figure_save_path}_{dataset_name}', save=save)


def main(task, model, dataset, num_epochs=100, lr=0.01, batch_size=1, enable_progress_bar=False):
    cluster = True
    dataset_path = '/data/f.caldas/gnn/datasets/' if cluster else 'datasets/'
    metrics_save_path = '/data/f.caldas/gnn/logs/' if cluster else 'logs/'
    model_save_path = '/data/f.caldas/gnn/models/' if cluster else 'models/'
    figure_save_path = '/data/f.caldas/gnn/figures/' if cluster else 'figures_local/'

    models = ['gcn', 'gin', 'gat', 'graphsage']
    nc_datasets = ['ba_shapes', 'ba_community', 'tree_cycles', 'tree_grid']
    gc_datasets = ['ba_2motif', 'mutag']
    lp_datasets = ['cora']
    save_figures = True
    stds = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    start = time.time()
    if task in ['nc', 'all']:
        nc_start = time.time()
        for std in stds:
            nc_std_start = time.time()
            datasets = []
            dataset = get_nc_dataset(dataset_path, 'ba_shapes', std_str=std, new=False)
            datasets.append(('ba_shapes', dataset))
            dataset = get_nc_dataset(dataset_path, 'tree_grid', std_str=std, new=False)
            datasets.append(('tree_grid', dataset))
            for model_name in models:
                nc_model_start = time.time()
                loggers = node_classification(model_name, datasets, metrics_save_path, model_save_path,
                                              num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                              enable_progress_bar=enable_progress_bar, std=std)

                save_path = f'{figure_save_path}{task}_{model_name}{std}'
                plot(loggers, save_path, save_figures)
                nc_model_end = time.time()
                nc_model_elapsed = (nc_model_end - nc_model_start) / 60
                print(f'{model_name} node classification took {nc_model_elapsed:.2f} minutes')

            nc_std_end = time.time()
            nc_std_elapsed = (nc_std_end - nc_std_start) / 60
            print(f'Node classification with std {std} took {nc_std_elapsed:.2f} minutes')

        nc_end = time.time()
        nc_elapsed = (nc_end - nc_start) / 60
        print(f'Node classification took {nc_elapsed:.2f} minutes')
    if task in ['gc', 'all']:
        gc_start = time.time()
        datasets = []
        data_list, num_classes = get_gc_dataset(dataset_path, 'mutag')
        datasets.append(('mutag', data_list, num_classes))
        for model_name in models:
            gc_model_start = time.time()
            loggers = graph_classification(model_name, datasets, metrics_save_path, model_save_path,
                                           num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                           enable_progress_bar=enable_progress_bar)
            save_path = f'{figure_save_path}{task}_{model_name}'
            #plot(loggers, save_path, save_figures)
            gc_model_end = time.time()
            gc_model_elapsed = (gc_model_end - gc_model_start) / 60
            print(f'{model_name} graph classification took {gc_model_elapsed:.2f} minutes')
        gc_end = time.time()
        gc_elapsed = (gc_end - gc_start) / 60
        print(f'Graph classification on MUTAG took {gc_elapsed:.2f} minutes')
        gc_start = time.time()
        for std in stds:
            datasets = []
            data_list, num_classes = get_gc_dataset(dataset_path, 'ba_2motif', std_str=std)
            datasets.append(('ba_2motif', data_list, num_classes))
            for model_name in models:
                gc_model_start = time.time()
                loggers = graph_classification(model_name, datasets, metrics_save_path, model_save_path,
                                               num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                               enable_progress_bar=enable_progress_bar, std=std)
                save_path = f'{figure_save_path}{task}_{model_name}{std}'
                plot(loggers, save_path, save_figures)
                gc_model_end = time.time()
                gc_model_elapsed = (gc_model_end - gc_model_start) / 60
                print(f'{model_name} graph classification took {gc_model_elapsed:.2f} minutes')
        gc_end = time.time()
        gc_elapsed = (gc_end - gc_start) / 60
        print(f'Graph classification on BA-2motif took {gc_elapsed:.2f} minutes')

    if task in ['lp', 'all']:
        lp_start = time.time()
        for std in stds:
            lp_std_start = time.time()
            datasets = []
            train_data, val_data, test_data = get_lp_dataset(dataset_path, 'cora')
            datasets.append(('cora', train_data, val_data, test_data))
            for model_name in models:
                lp_model_start = time.time()
                loggers = link_prediction(model_name, datasets, metrics_save_path, model_save_path,
                                          num_epochs=num_epochs, lr=lr, batch_size=batch_size,
                                          enable_progress_bar=enable_progress_bar, std=std)
                save_path = f'{figure_save_path}{task}_{model_name}{std}'
                plot(loggers, save_path, save_figures)
                lp_model_end = time.time()
                lp_model_elapsed = (lp_model_end - lp_model_start) / 60
                print(f'{model_name} link prediction took {lp_model_elapsed:.2f} minutes')
            lp_std_end = time.time()
            lp_std_elapsed = (lp_std_end - lp_std_start) / 60
            print(f'Link prediction with std {std} took {lp_std_elapsed:.2f} minutes')
        lp_end = time.time()
        lp_elapsed = (lp_end - lp_start) / 60
        print(f'Link prediction took {lp_elapsed:.2} minutes')

    end = time.time()
    elapsed = (end - start) / 60
    print(f'Total time elapsed: {elapsed:.2f} minutes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN models on node and graph classification datasets.')
    parser.add_argument('--task', type=str, default='all',
                        help='Task to perform: nc (node classification) or gc (graph classification).')
    parser.add_argument('--model', type=str, default='all', help='Model to use: gcn, gin, gat, graphsage, or all.')
    parser.add_argument('--dataset', type=str, default='all',
                        help='Name of the dataset to use: ba_shapes(nc), ba_community(nc), tree_cycles(nc), tree_grid(nc), ba_2motif(gc), mutag(gc), or all.')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the data loader.')

    args = parser.parse_args()
    print(args)
    main(args.task, args.model, args.dataset, args.num_epochs, args.lr, args.batch_size)
