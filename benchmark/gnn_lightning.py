import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
import torch_geometric as pyg
import pytorch_lightning as pl


class GNN(pl.LightningModule):
    def __init__(self, model=None, optimizer=None, criterion=None, is_multiclass=None, is_lp=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.is_multiclass = is_multiclass
        self.is_lp = is_lp

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None,
                num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, edge_label_index=None):
        return self.model(x, edge_index, edge_weight, edge_attr, batch, batch_size, num_sampled_nodes_per_hop,
                          num_sampled_edges_per_hop)

    def _step(self, data):
        if ('edge_weight' in data) and ('edge_attr' in data) and ('batch' in data):
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, data.edge_weight, data.edge_attr, data.batch,
                            edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, data.edge_weight, data.edge_attr, data.batch)
        elif ('edge_weight' in data) and ('edge_attr' in data):
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, data.edge_weight, data.edge_attr,
                            edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, data.edge_weight, data.edge_attr)
        elif ('edge_weight' in data) and ('batch' in data):
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, data.edge_weight, batch=data.batch,
                            edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, data.edge_weight, batch=data.batch)
        elif ('edge_attr' in data) and ('batch' in data):
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch, edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        elif 'edge_weight' in data:
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, data.edge_weight, edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, data.edge_weight)
        elif 'edge_attr' in data:
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, edge_attr=data.edge_attr, edge_label_index=data.edge_label)
            return self(data.x, data.edge_index, edge_attr=data.edge_attr)
        elif 'batch' in data:
            if ('edge_label_index' in data):
                return self(data.x, data.edge_index, batch=data.batch, edge_label_index=data.edge_label_index)
            return self(data.x, data.edge_index, batch=data.batch)
        elif 'edge_label_index' in data:
            return self(data.x, data.edge_index, edge_label_index=data.edge_label_index)
        return self(data.x, data.edge_index)

    def configure_optimizers(self):
        return self.optimizer

    def _compute_metrics(self, y_true, y_pred):
        """
        Computes accuracy, precision, recall, F1 score and confusion matrix metrics for a classification task.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=np.nan)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=np.nan)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=np.nan)
        cm = None  # confusion_matrix(y_true, y_pred)
        return accuracy, precision, recall, f1, cm

    def _compute_and_log_metrics(self, y_pred, y_true, phase):
        """Computes scores, loss, metrics, and logs them for the given phase."""
        loss = self.criterion(y_pred if self.is_multiclass else y_pred.view_as(y_true),
                              y_true if self.is_multiclass else y_true.float())
        if self.is_multiclass:
            y_pred = y_pred.argmax(dim=-1)
        else:
            y_pred = y_pred.sigmoid().round().detach()
        acc, prec, rec, f1, cm = self._compute_metrics(y_true.cpu(), y_pred.cpu())

        self.log(f"{phase}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log(f"{phase}_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log(f"{phase}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log(f"{phase}_precision", prec, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        self.log(f"{phase}_recall", rec, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1)

        return loss


class NC_GNN(GNN):
    def __init__(self, model=None, optimizer=None, criterion=None, is_multiclass=None, is_lp=None):
        super().__init__(model, optimizer, criterion, is_multiclass, is_lp)

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None,
                num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, edge_label_index=None):
        output = super().forward(x, edge_index, edge_weight, edge_attr, batch, batch_size, num_sampled_nodes_per_hop,
                                 num_sampled_edges_per_hop)
        return output

    def training_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output[batch.train_mask], batch.y[batch.train_mask], 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output[batch.val_mask], batch.y[batch.val_mask], 'val')
        return loss

    def test_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output[batch.test_mask], batch.y[batch.test_mask], 'test')
        return loss


class GC_GNN(GNN):
    def __init__(self, model=None, optimizer=None, criterion=None, is_multiclass=None, is_lp=None, out_channels=None,
                 pooling_type=None):
        super().__init__(model, optimizer, criterion, is_multiclass, is_lp)
        self.pooling_layer = self._pooling_layer(pooling_type)
        self.lin = torch.nn.Linear(model.out_channels, out_channels)

    def _pooling_layer(self, pooling_type):
        if pooling_type == 'mean':
            return pyg.nn.global_mean_pool
        elif pooling_type == 'max':
            return pyg.nn.global_max_pool
        elif pooling_type == 'add':
            return pyg.nn.global_add_pool
        else:
            raise ValueError(f'Pooling type {pooling_type} is not supported.')

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None,
                num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, edge_label_index=None):
        output = super().forward(x, edge_index, edge_weight, edge_attr, batch, batch_size, num_sampled_nodes_per_hop,
                                 num_sampled_edges_per_hop)
        output = self.pooling_layer(output, batch)
        output = self.lin(output)
        return output

    def training_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.y, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.y, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.y, 'test')
        return loss


class LP_GNN(GNN):
    def __init__(self, model=None, optimizer=None, criterion=None, is_multiclass=None, is_lp=None, decoder=None):
        super().__init__(model, optimizer, criterion, is_multiclass, is_lp)
        self.decoder = decoder

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None,
                num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, edge_label_index=None):
        output = super().forward(x, edge_index, edge_weight, edge_attr, batch, batch_size, num_sampled_nodes_per_hop,
                                 num_sampled_edges_per_hop)
        output = self.decoder(output, edge_label_index, sigmoid=False)
        output = torch.nn.functional.relu(output)
        return output

    def training_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.edge_label, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.edge_label, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        output = super()._step(batch)
        loss = self._compute_and_log_metrics(output, batch.edge_label, 'test')
        return loss
