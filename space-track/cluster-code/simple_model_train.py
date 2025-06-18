import argparse
import time

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch_geometric.loader import DataLoader as GDataLoader
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import pytorch_lightning as L

import matplotlib.pyplot as plt
import seaborn as sns

from dataset_splitter_converter_sampler import sample_new, load_existing
from dynamic_graph_dataset import DTDGLinkDataset
from lightning_model import STGNN


from sklearn.metrics import f1_score, confusion_matrix


class SimpleModel(L.LightningModule):
    def __init__(self, loss_fn, loss_name='bce', pos_weight = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.loss_name = loss_name
        self.pos_weight = pos_weight
        self.automatic_optimization = False

    def _disjoint_concatenated_eis(self, concatenated_ei, num_nodes_per_graph, num_graphs):
        graphs_edge_indexes = []

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
            graph_edge_index = concatenated_ei[:, mask] - min_node_id  # Normalize to start at 0
            graphs_edge_indexes.append(graph_edge_index)

        return graphs_edge_indexes

    def _compute_score(self, last_graph_ei, target_ei, pos=True):
        """
        Computes a tensor of shape E2 with 0s and 1s.

        Parameters:
        last_graph_ei (torch.Tensor): Tensor of shape 2xE1 where E1 is the number of edges in the last graph.
        target_ei (torch.Tensor): Tensor of shape 2xE2 where E2 is the number of edges in the target graph.
        pos (bool): If True, return 1 if an edge from target_ei is present in last_graph_ei.
                    If False, return 0 if an edge from target_ei is not present in last_graph_ei.

        Returns:
        torch.Tensor: Tensor of shape E2 with 0s and 1s.
        """
        # # Convert the edge lists to sets of tuples for easy comparison
        # last_graph_edges = set(map(tuple, last_graph_ei.t().tolist()))
        # target_edges = list(map(tuple, target_ei.t().tolist()))
        #
        # # Compute the result based on the pos flag
        # result = [(1 if edge in last_graph_edges else 0) if pos else (0 if edge in last_graph_edges else 1) for edge in target_edges]
        #
        # # Convert the result to a tensor and return
        # return torch.tensor(result, dtype=torch.int64)
        # Expand dimensions for broadcasting
        last_graph_ei_expanded = last_graph_ei.unsqueeze(2)  # Shape: [2, E1, 1]
        target_ei_expanded = target_ei.unsqueeze(1)  # Shape: [2, 1, E2]

        # Compare edges using broadcasting
        matches = (last_graph_ei_expanded == target_ei_expanded).all(dim=0)  # Shape: [E1, E2]

        # Check if each target edge is present in last_graph_ei
        result = matches.any(dim=0).to(torch.float64)  # Shape: [E2]

        # Adjust result based on pos flag
        if not pos:
            result = 1 - result

        return result

    def _forward(self, batch, num_graphs, num_nodes_per_graph):
        batch_pos_score = []
        batch_neg_score = []
        for window_id in range(num_graphs):
            window = batch[window_id]
            last_graph_ei = self._disjoint_concatenated_eis(window.edge_index['input_pos_ei'], num_nodes_per_graph,
                                                            num_graphs)[-1]
            pos_score = self._compute_score(last_graph_ei, window.edge_index['target_pos_ei'], pos=True)
            neg_score = self._compute_score(last_graph_ei, window.edge_index['target_neg_ei'], pos=False)
            batch_pos_score.append(pos_score)
            batch_neg_score.append(neg_score)

        return torch.concat(batch_pos_score, dim=0), torch.concat(batch_neg_score, dim=0)

    def _compute_metrics(self, pos_score, neg_score):
        y_true = torch.cat([
            torch.ones(pos_score.size(0), device=self.device),
            torch.zeros(neg_score.size(0), device=self.device)
        ]).detach().cpu().numpy()

        y_pred = torch.cat([pos_score, neg_score]).detach().cpu().numpy() > 0.5

        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        acc = ((pos_score >= 0.5).float().sum() + (neg_score < 0.5).float().sum()) / (
                pos_score.size(0) + neg_score.size(0))

        return acc, f1, cm

    def _process_batch(self, batch):
        """Unpacks the batch and processes the inputs through the encoder."""
        pos_score, neg_score = self._forward(batch, batch.num_graphs, int(batch.num_nodes / batch.num_graphs))
        return pos_score, neg_score

    def _compute_and_log_metrics(self, phase, pos_score, neg_score, pos_weight=None):
        """Computes scores, loss, metrics, and logs them for the given phase."""

        loss = self.loss_fn(pos_score, neg_score, device=self.device, loss_name=self.loss_name,
                            pos_weight=self.pos_weight)
        acc, f1, cm = self._compute_metrics(pos_score, neg_score)
        tn, fp, fn, tp = cm.ravel()

        self.log(f"{phase}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_tn", tn, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_fp", fp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_fn", fn, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)
        self.log(f"{phase}_tp", tp, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=8)

        return loss

    def training_step(self, batch, batch_idx):
        pos_score, neg_score = self._process_batch(batch)
        return self._compute_and_log_metrics("train", pos_score, neg_score, self.pos_weight)

    def validation_step(self, batch, batch_idx):
        pos_score, neg_score = self._process_batch(batch)
        return self._compute_and_log_metrics("val", pos_score, neg_score, self.pos_weight)

    def test_step(self, batch, batch_idx):
        pos_score, neg_score = self._process_batch(batch)
        return self._compute_and_log_metrics("test", pos_score, neg_score)

    def configure_optimizers(self):
        return None


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


def main():
    parser = argparse.ArgumentParser(
        description='Train a spatio-temporal graph neural network on the space-track dataset')
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
            data_folder_location=args.data_folder_location, weight=-1)

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

    losses = ['bce', 'hinge', 'weighted_bce']
    loss_name = losses[0]  # args.loss_name
    sm = SimpleModel(loss_fn=compute_loss, loss_name=loss_name)
    print(sm)
    max_epochs = 100  # args.max_epochs
    wb = False
    if wb:
        # Log in to your W&B account
        import wandb

        wandb.login()
        logger = WandbLogger(project="thesis-space")
    else:
        logger = CSVLogger(save_dir="logs",
                           name=f'simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}',
                           version=1)

    trainer = L.Trainer(accelerator='auto', max_epochs=max_epochs, log_every_n_steps=5, logger=logger,
                        enable_progress_bar=False)
    trainer.fit(model=sm, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_dataset = DTDGLinkDataset(test_pos_data, test_neg_data, window=window, horizon=horizon)
    print(len(test_dataset))
    test_loader = GDataLoader(test_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

    trainer.test(sm, dataloaders=test_loader)

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

        figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}.pdf'
        save_figure(fig, figure_savepath, save)

    elif plot_type == 'acc-loss-f1-cm':
        fig_loss, ax1 = plt.subplots()
        plot_loss(ax1, train_loss_per_epoch, val_loss_per_epoch)
        fig_loss.tight_layout()

        loss_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-loss.pdf'
        save_figure(fig_loss, loss_figure_savepath, save)

        fig_acc, ax2 = plt.subplots()
        plot_accuracy(ax2, train_acc_per_epoch, val_acc_per_epoch)
        fig_acc.tight_layout()

        acc_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-accuracy.pdf'
        save_figure(fig_acc, acc_figure_savepath, save)

        fig_f1, ax3 = plt.subplots()
        plot_f1(ax3, train_f1_per_epoch, val_f1_per_epoch)
        fig_f1.tight_layout()

        f1_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-f1.pdf'
        save_figure(fig_f1, f1_figure_savepath, save)

        fig_train_cm_metrics, ax4 = plt.subplots()
        plot_cm_metrics(ax4, train_tn_per_epoch, train_fp_per_epoch, train_fn_per_epoch, train_tp_per_epoch)

        cm_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-train-cm-metrics.pdf'
        save_figure(fig_train_cm_metrics, cm_figure_savepath, save)

        fig_val_cm_metrics, ax5 = plt.subplots()
        plot_cm_metrics(ax5, val_tn_per_epoch, val_fp_per_epoch, val_fn_per_epoch, val_tp_per_epoch)

        cm_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-val-cm-metrics.pdf'
        save_figure(fig_val_cm_metrics, cm_figure_savepath, save)

        fig_train_cm, ax6 = plt.subplots()
        plot_confusion_matrix(ax6, train_tn_per_epoch, train_fp_per_epoch, train_fn_per_epoch, train_tp_per_epoch)

        cm_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-train-cm.pdf'
        save_figure(fig_train_cm, cm_figure_savepath, save)

        fig_val_cm, ax7 = plt.subplots()
        plot_confusion_matrix(ax7, val_tn_per_epoch, val_fp_per_epoch, val_fn_per_epoch, val_tp_per_epoch)

        cm_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-val-cm.pdf'
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
        train_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-train.pdf'
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
        val_figure_savepath = f'figures/simplemodel-{loss_name}-w-{window}-e-{max_epochs}-b-{batch_size}-val.pdf'
        save_figure(fig_val, val_figure_savepath, save)
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Script completed in {elapsed_time:.2f} minutes")


if __name__ == '__main__':
    main()
