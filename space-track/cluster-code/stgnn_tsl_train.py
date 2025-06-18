import argparse
import time

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GDataLoader
import torch_geometric.nn as nng
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as L

import matplotlib.pyplot as plt
import seaborn as sns

from dataset_splitter_converter_sampler import sample_new, load_existing, import_satellite_graph, temporal_signal_split, \
    sample_negative_edges, to_tensor_data
from dynamic_graph_dataset import DTDGLinkDataset
from layers import TimeSpaceModel
from lightning_model import STGNN


# def compute_loss(pos_score, neg_score, device=None):
#     scores = torch.cat([pos_score, neg_score])
#     labels = torch.cat(
#         [torch.ones(pos_score.shape[0], device=device), torch.zeros(neg_score.shape[0], device=device)]
#     ).double()
#     return F.binary_cross_entropy(scores, labels)


def compute_loss(pos_score, neg_score, device=None, loss_name='bce', pos_weight=None):
    """
    Computes loss based on the specified loss name.

    Args:
        pos_score (torch.Tensor): Positive class scores.
        neg_score (torch.Tensor): Negative class scores.
        device (torch.device, optional): Device for tensor operations.
        loss_name (str, optional): Type of loss function to compute ('bce', 'hinge', 'weighted_bce').
        class_weights (torch.Tensor, optional): Class weights for weighted BCE loss, expected as [weight_pos, weight_neg].

    Returns:
        torch.Tensor: Computed loss.
    """
    # Concatenate scores and labels
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat(
        [torch.ones(pos_score.shape[0], device=device), torch.zeros(neg_score.shape[0], device=device)]
    ).double()

    if loss_name == 'bce':
        # Binary Cross Entropy Loss
        return F.binary_cross_entropy(scores, labels)

    elif loss_name == 'hinge':
        # Hinge Loss
        hinge_labels = 2 * labels - 1  # Convert labels to {-1, 1} for hinge loss
        return torch.mean(torch.clamp(1 - scores * hinge_labels, min=0))
    elif loss_name == 'weighted_bce':
        # Convert scores (probabilities) to logits
        logits = torch.log(scores / (1 - scores))  # Inverse of the sigmoid function

        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    else:
        raise ValueError(f"Unsupported loss_name: {loss_name}. Choose from 'bce', 'hinge', 'weighted_bce'.")


# Helper function to save figures
def save_figure(fig, save_path, save):
    if save:
        fig.savefig(save_path, format='pdf')


# Helper function to plot training and validation loss
def plot_loss(ax, train_loss, val_loss):
    color = 'tab:red'
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss', color=color)
    ax.plot(train_loss, color=color, label='Train Loss')
    ax.plot(val_loss, color='tab:orange', linestyle='--', label='Val Loss')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))


# Helper function to plot training and validation accuracy
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


# Line plot of the confusion matrix metrics over epochs
def plot_cm_metrics(ax, tn, fp, fn, tp):
    metrics = [tn, fp, fn, tp]
    for m_name, m_values in zip(['True Negative', 'False Positive', 'False Negative', 'True Positive'], metrics):
        ax.plot(range(m_values.shape[0]), m_values, label=m_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Values')
    ax.legend()
    ax.grid(True)


# Compute summed values for each confusion matrix metric
def plot_confusion_matrix(ax, tn, fp, fn, tp):
    summed_values = {
        'True Negative': tn.sum(),
        'False Positive': fp.sum(),
        'False Negative': fn.sum(),
        'True Positive': tp.sum()
    }

    # Create a confusion matrix from the summed values
    matrix = np.array([[summed_values['True Negative'], summed_values['False Positive']],
                       [summed_values['False Negative'], summed_values['True Positive']]])

    # convert matrix to integers
    matrix = matrix.astype(int)

    # Plot confusion matrix heatmap
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         matrix.flatten() / np.sum(matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(matrix, annot=labels, fmt="", cmap='Blues')
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')


def save_model(model, time_module, time_layers, space_module, space_layers, hidden_size, output_size, window,
               max_epochs, batch_size, **kwargs):
    model_savepath = f'models/stgnn-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}'
    model_savepath = kwargs.get('data_folder_location', '') + model_savepath
    torch.save(model, model_savepath)


def main():
    parser = argparse.ArgumentParser(
        description='Train a spatio-temporal graph neural network on the space-track dataset')
    parser.add_argument('--weight', type=int, default=-1,
                        help='The weight of the edge. Chose index from ["r_dist", "ct_dist", "it_dist", "dist"]')
    parser.add_argument('--loss_name', type=str, default='bce',help='The loss function to use. Choose from ["bce", "hinge", "weighted_bce"]')
    parser.add_argument('--window', type=int, default=2, help='The window size for the model')

    parser.add_argument('--hidden_size', type=int, default=16, help='The hidden size for the model')
    parser.add_argument('--output_size', type=int, default=8, help='The output size for the model')
    parser.add_argument('--time_module', type=str, default='gru',
                        help='The time module to use. Chose from ["gru", "lstm", "multigru", "multilstm", "graphconvgru", "graphconvlstm", "dcrnn", "evolvegcn", "tranformer", "cnn", "gcnn", "att", "tcn", "condtcn", "stcn", None]')
    parser.add_argument('--time_layers', type=int, default=1, help='The number of layers for the time module')
    parser.add_argument('--space_module', type=str, default='gcn',
                        help='The space module to use. Choose from ["gcn", "diffconv", "gat", "sage", "gatconv", None]')
    parser.add_argument('--space_layers', type=int, default=1, help='The number of layers for the space module')
    parser.add_argument('--dec', type=int, default=0,
                        help='The decoder to use for the model. Choose index from ["dot", "concat", "hadamard", "l1", "l2", "avg", "mlp"]')
    parser.add_argument('--max_epochs', type=int, default=1, help='The maximum number of epochs to train the model')
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

    #options_sample_leos = (True, 0.25, 'leo4', True)
    # options_sample_alt_e = (True, 1.0, 500, 520, 0.25, False)
    # nodes_df, edges_df, timestamps = import_satellite_graph(options_sample_leos=options_sample_leos,
    #                                                         data_folder_location=data_folder_location,
    #                                                         weight=args.weight)
    #
    # print(timestamps)
    # print(timestamps.size)
    #
    # train_timestamps, val_timestamps, test_timestamps = temporal_signal_split(timestamps)
    #
    # train_edges_df = edges_df[edges_df['timestamp'].isin(train_timestamps)]
    # val_edges_df = edges_df[edges_df['timestamp'].isin(val_timestamps)]
    # test_edges_df = edges_df[edges_df['timestamp'].isin(test_timestamps)]
    #
    # print(
    #     f"Number of train timestamps: {train_timestamps.shape[0]}\n"
    #     f"Number of validation timestamps: {val_timestamps.shape[0]}\n"
    #     f"Number of test timestamps: {test_timestamps.shape[0]}"
    # )
    #
    # nodes = list(nodes_df.index.unique())
    # num_nodes = len(nodes)
    # node_index = {node: i for i, node in enumerate(nodes)}
    #
    # # Train set
    # train_pos_edges_df = train_edges_df[['source', 'target', 'weight', 'timestamp']]
    #
    # print("Sampling negative training edges...")
    # train_neg_edges_df = sample_negative_edges(nodes, train_pos_edges_df, train_timestamps)
    #
    # print("Converting to pyg tensor data...")
    # train_pos_data = to_tensor_data(train_timestamps, num_nodes, node_index, nodes_df, train_pos_edges_df)
    # train_neg_data = to_tensor_data(train_timestamps, num_nodes, node_index, nodes_df, train_neg_edges_df)
    #
    # print(
    #     f"Number of total positive edges in training set: {train_pos_edges_df.shape[0]}\n"
    #     f"Number of total negative edges in training set: {train_neg_edges_df.shape[0]}\n"
    # )
    #
    # # Validation set
    # val_pos_edges_df = val_edges_df[['source', 'target', 'weight', 'timestamp']]
    #
    # print("Sampling negative validation edges...")
    # val_neg_edges_df = sample_negative_edges(nodes, val_pos_edges_df, val_timestamps)
    #
    # print("Converting to pyg tensor data...")
    # val_pos_data = to_tensor_data(val_timestamps, num_nodes, node_index, nodes_df, val_pos_edges_df)
    # val_neg_data = to_tensor_data(val_timestamps, num_nodes, node_index, nodes_df, val_neg_edges_df)
    #
    # print(
    #     f"Number of total positive edges in validation set: {val_pos_edges_df.shape[0]}\n"
    #     f"Number of total negative edges in validation set: {val_neg_edges_df.shape[0]}\n"
    # )
    #
    # # Test set
    # test_pos_edges_df = test_edges_df[['source', 'target', 'weight', 'timestamp']]
    #
    # print("Sampling negative testing edges...")
    # test_neg_edges_df = sample_negative_edges(nodes, test_pos_edges_df, test_timestamps)
    #
    # print("Converting to pyg tensor data...")
    # test_pos_data = to_tensor_data(test_timestamps, num_nodes, node_index, nodes_df, test_pos_edges_df)
    # test_neg_data = to_tensor_data(test_timestamps, num_nodes, node_index, nodes_df, test_neg_edges_df)
    #
    #
    # print(
    #     f"Number of total positive edges in test set: {test_pos_edges_df.shape[0]}\n"
    #     f"Number of total negative edges in test set: {test_neg_edges_df.shape[0]}\n"
    # )

    # ---------------------------------------- Defining the dataset hyperparemeters ------------------------------------

    window = 128  # args.window  # 16  # should be < len(train_pos_data[0]) train_pos_data[0] == pos_xs
    horizon = 1
    shuffle = False
    batch_size = 32

    train_dataset = DTDGLinkDataset(train_pos_data, train_neg_data, window=window, horizon=horizon)
    train_loader = GDataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)
    print(len(train_dataset))
    batch = next(iter(train_loader))
    print(batch)

    val_dataset = DTDGLinkDataset(val_pos_data, val_neg_data, window=window, horizon=horizon)
    val_loader = GDataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)
    print(len(val_dataset))
    batch = next(iter(val_loader))
    print(batch)

    # --------------------------------------------- Training the model -------------------------------------------------
    print("Training the model...")
    num_nodes = len(list(nodes_df.index.unique()))
    input_size = len(nodes_df.columns) - 1
    hidden_size = 64  # args.hidden_size
    output_size = 128  # args.output_size
    time_layers = 1  # args.time_layers
    space_layers = 1  # args.space_layers

    time_modules = ['gru', 'lstm', 'multigru', 'multilstm', 'graphconvgru', 'graphconvlstm', 'dcrnn', 'evolvegcn',
                    'tranformer', 'cnn', 'gcnn', 'att', 'tcn', 'condtcn', 'stcn', None]
    time_module = time_modules[9]  # args.time_module
    space_modules = ['gcn', 'diffconv', 'gat', 'sage', 'gatconv', None]
    space_module = space_modules[0]  # args.space_module
    model = TimeSpaceModel(time_module=time_module, space_module=space_module, input_size=input_size,
                           output_size=output_size, hidden_size=hidden_size, time_layers=time_layers,
                           space_layers=space_layers, window=window,
                           horizon=horizon, num_nodes=num_nodes, batch_size=batch_size).double()

    clf = nng.InnerProductDecoder()  # args.dec
    losses = ['bce', 'hinge', 'weighted_bce']
    loss_name = losses[0]  # args.loss_name
    stgnn = STGNN(encoder=model, clf=clf, loss_fn=compute_loss, loss_name=loss_name)
    print(stgnn)

    max_epochs = 100  # args.max_epochs
    wb = False
    if wb:
        # Log in to your W&B account
        import wandb

        wandb.login()
        logger = WandbLogger(project="thesis-space")
    else:
        logger = CSVLogger(save_dir="logs",
                           name=f'stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}',
                           version=1)

    trainer = L.Trainer(accelerator='gpu', max_epochs=max_epochs, log_every_n_steps=5, logger=logger,
                        enable_progress_bar=False)
    trainer.fit(model=stgnn, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_dataset = DTDGLinkDataset(test_pos_data, test_neg_data, window=window, horizon=horizon)
    test_loader = GDataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    print(len(test_dataset))
    trainer.test(stgnn, dataloaders=test_loader)

    # -------------------------------------------- Save the model ------------------------------------------------------
    save_model(stgnn, time_module, time_layers, space_module, space_layers, hidden_size, output_size, window,
               max_epochs, batch_size,
               data_folder_location=args.data_folder_location)

    # --------------------------------------- Plotting Training Statistics ---------------------------------------------
    print("Plotting Training Statistics...")
    save = True
    # Read the training statistics from the CSV file
    train_stats = pd.read_csv(logger.log_dir + '/metrics.csv')

    # Extract loss and accuracy per epoch for both training and validation
    train_loss_per_epoch = train_stats['train_loss'].dropna().reset_index(drop=True)
    train_acc_per_epoch = train_stats['train_accuracy'].dropna().reset_index(drop=True)

    val_loss_per_epoch = train_stats['val_loss'].dropna().reset_index(drop=True)
    val_acc_per_epoch = train_stats['val_accuracy'].dropna().reset_index(drop=True)

    train_f1_per_epoch = train_stats['train_f1'].dropna().reset_index(drop=True)
    val_f1_per_epoch = train_stats['val_f1'].dropna().reset_index(drop=True)

    train_tn_per_epoch = train_stats['train_tn'].dropna().reset_index(drop=True)
    train_fp_per_epoch = train_stats['train_fp'].dropna().reset_index(drop=True)
    train_fn_per_epoch = train_stats['train_fn'].dropna().reset_index(drop=True)
    train_tp_per_epoch = train_stats['train_tp'].dropna().reset_index(drop=True)

    val_tn_per_epoch = train_stats['val_tn'].dropna().reset_index(drop=True)
    val_fp_per_epoch = train_stats['val_fp'].dropna().reset_index(drop=True)
    val_fn_per_epoch = train_stats['val_fn'].dropna().reset_index(drop=True)
    val_tp_per_epoch = train_stats['val_tp'].dropna().reset_index(drop=True)

    plot_types = ['one-for-all', 'acc-loss-f1-cm', 'train-val']
    plot_type = plot_types[1]

    if plot_type == 'one-for-all':
        fig, ax1 = plt.subplots()
        plot_loss(ax1, train_loss_per_epoch, val_loss_per_epoch)
        ax2 = ax1.twinx()
        plot_accuracy(ax2, train_acc_per_epoch, val_acc_per_epoch)
        ax3 = ax1.twinx()
        plot_f1(ax3, train_f1_per_epoch, val_f1_per_epoch)
        fig.tight_layout()

        figure_savepath = f'figures/stgnn-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}.pdf'
        save_figure(fig, figure_savepath, save)

    elif plot_type == 'acc-loss-f1-cm':
        fig_loss, ax1 = plt.subplots()
        plot_loss(ax1, train_loss_per_epoch, val_loss_per_epoch)
        fig_loss.tight_layout()

        loss_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-loss.pdf'
        save_figure(fig_loss, loss_figure_savepath, save)

        fig_acc, ax2 = plt.subplots()
        plot_accuracy(ax2, train_acc_per_epoch, val_acc_per_epoch)
        fig_acc.tight_layout()

        acc_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-accuracy.pdf'
        save_figure(fig_acc, acc_figure_savepath, save)

        fig_f1, ax3 = plt.subplots()
        plot_f1(ax3, train_f1_per_epoch, val_f1_per_epoch)
        fig_f1.tight_layout()

        f1_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-f1.pdf'
        save_figure(fig_f1, f1_figure_savepath, save)

        fig_train_cm_metrics, ax4 = plt.subplots()
        plot_cm_metrics(ax4, train_tn_per_epoch, train_fp_per_epoch, train_fn_per_epoch, train_tp_per_epoch)

        cm_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-train-cm-metrics.pdf'
        save_figure(fig_train_cm_metrics, cm_figure_savepath, save)

        fig_val_cm_metrics, ax5 = plt.subplots()
        plot_cm_metrics(ax5, val_tn_per_epoch, val_fp_per_epoch, val_fn_per_epoch, val_tp_per_epoch)

        cm_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-val-cm-metrics.pdf'
        save_figure(fig_val_cm_metrics, cm_figure_savepath, save)

        fig_train_cm, ax6 = plt.subplots()
        plot_confusion_matrix(ax6, train_tn_per_epoch, train_fp_per_epoch, train_fn_per_epoch, train_tp_per_epoch)

        cm_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-train-cm.pdf'
        save_figure(fig_train_cm, cm_figure_savepath, save)

        fig_val_cm, ax7 = plt.subplots()
        plot_confusion_matrix(ax7, val_tn_per_epoch, val_fp_per_epoch, val_fn_per_epoch, val_tp_per_epoch)

        cm_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-val-cm.pdf'
        save_figure(fig_val_cm, cm_figure_savepath, save)

    elif plot_type == 'train-val':
        # Plotting Training Statistics
        fig_train, ax1 = plt.subplots()

        # Plot training loss
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Training Loss', color=color)
        ax1.plot(train_loss_per_epoch, color=color, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for training accuracy
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Training Accuracy', color=color)
        ax2.plot(train_acc_per_epoch, color=color, label='Train Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        # Combine legends from both y-axes
        fig_train.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        fig_train.tight_layout()

        # Save the training figure based on conditions
        train_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-train.pdf'
        save_figure(fig_train, train_figure_savepath, save)

        # Plotting Validation Statistics
        fig_val, ax3 = plt.subplots()

        # Plot validation loss
        color = 'tab:orange'
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Validation Loss', color=color)
        ax3.plot(val_loss_per_epoch, color=color, linestyle='--', label='Val Loss')
        ax3.tick_params(axis='y', labelcolor=color)

        # Create a second y-axis for validation accuracy
        ax4 = ax3.twinx()
        color = 'tab:cyan'
        ax4.set_ylabel('Validation Accuracy', color=color)
        ax4.plot(val_acc_per_epoch, color=color, linestyle='--', label='Val Accuracy')
        ax4.tick_params(axis='y', labelcolor=color)

        # Combine legends from both y-axes
        fig_val.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

        fig_val.tight_layout()

        # Save the validation figure based on conditions
        val_figure_savepath = f'figures/stgnn-{loss_name}-{time_module}-{time_layers}-{space_module}-{space_layers}-h-{hidden_size}-o-{output_size}-w-{window}-e-{max_epochs}-b-{batch_size}-val.pdf'
        save_figure(fig_val, val_figure_savepath, save)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script completed in {elapsed_time:.2f} minutes")


if __name__ == '__main__':
    main()
