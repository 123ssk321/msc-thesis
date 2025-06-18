import itertools
from typing import Any
import torch
import pytorch_lightning as L
from sklearn.metrics import f1_score, confusion_matrix


class STGNN(L.LightningModule):
    def __init__(self, encoder, clf, loss_fn, loss_name=None, pos_weight=None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.clf = clf
        self.loss_fn = loss_fn
        self.loss_name = loss_name
        self.pos_weight = pos_weight

    def _unpack_batch(self, batch):
        x, edge_index, edge_weight = batch.input.pos_x, batch.edge_index['input_pos_ei'], \
            batch.edge_weight['input_pos_w']
        target_pos_ei, target_neg_ei = batch.edge_index['target_pos_ei'], batch.edge_index['target_neg_ei']
        # print(f'x: {x.shape}')
        # print(f'edge_index: {edge_index.shape}')
        # print(f'edge_weight: {edge_weight.shape}')
        # print(f'target_pos_ei: {target_pos_ei.shape}')
        # print(f'target_neg_ei: {target_neg_ei.shape}')
        return x, edge_index, edge_weight, target_pos_ei, target_neg_ei

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

    # def training_step(self, batch, batch_idx):
    #     x, edge_index, edge_weight, target_pos_edge_index, target_neg_edge_index = self._unpack_batch(batch)
    #     h = self.encoder(x, edge_index, edge_weight)
    #
    #     pos_score = self.clf(h.squeeze().squeeze(), target_pos_edge_index)
    #     neg_score = self.clf(h.squeeze().squeeze(), target_neg_edge_index)
    #
    #     loss = self.loss_fn(pos_score, neg_score, device=self.device)
    #     acc, f1, cm = self._compute_metrics(pos_score, neg_score)
    #     tn, fp, fn, tp = cm.ravel()
    #     self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("train_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_tn", tn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_fp", fp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_fn", fn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("train_tp", tp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, edge_index, edge_weight, target_pos_edge_index, target_neg_edge_index = self._unpack_batch(batch)
    #     h = self.encoder(x, edge_index, edge_weight)
    #
    #     pos_score = self.clf(h.squeeze().squeeze(), target_pos_edge_index)
    #     neg_score = self.clf(h.squeeze().squeeze(), target_neg_edge_index)
    #
    #     loss = self.loss_fn(pos_score, neg_score, device=self.device)
    #     acc, f1, cm = self._compute_metrics(pos_score, neg_score)
    #     tn, fp, fn, tp = cm.ravel()
    #
    #     self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("val_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("val_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_tn", tn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_fp", fp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_fn", fn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("val_tp", tp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return loss
    #
    # def test_step(self, batch, batch_idx):
    #     x, edge_index, edge_weight, target_pos_edge_index, target_neg_edge_index = self._unpack_batch(batch)
    #     h = self.encoder(x, edge_index, edge_weight)
    #
    #     pos_score = self.clf(h.squeeze().squeeze(), target_pos_edge_index)
    #     neg_score = self.clf(h.squeeze().squeeze(), target_neg_edge_index)
    #
    #     loss = self.loss_fn(pos_score, neg_score, device=self.device)
    #     acc, f1, cm = self._compute_metrics(pos_score, neg_score)
    #     tn, fp, fn, tp = cm.ravel()
    #     self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("test_accuracy", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=7)
    #     self.log("test_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("test_tn", tn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("test_fp", fp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("test_fn", fn, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #     self.log("test_tp", tp, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    #
    #     return loss

    def _process_batch(self, batch):
        """Unpacks the batch and processes the inputs through the encoder."""
        x, edge_index, edge_weight, target_pos_edge_index, target_neg_edge_index = self._unpack_batch(batch)
        h = self.encoder(x, edge_index, edge_weight, batch.num_graphs)
        return h, target_pos_edge_index, target_neg_edge_index

    def _compute_and_log_metrics(self, phase, h, target_pos_edge_index, target_neg_edge_index, pos_weight=None):
        """Computes scores, loss, metrics, and logs them for the given phase."""
        #print(f'h: {h.shape}')
        #print(f'target_pos_edge_index: {target_pos_edge_index.shape}')
        #print(f'target_neg_edge_index: {target_neg_edge_index.shape}')
        #print(f'target_pos_edge_index: {target_pos_edge_index.max()}')
        pos_score = self.clf(h, target_pos_edge_index)
        neg_score = self.clf(h, target_neg_edge_index)

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

    def forward(self, x, edge_index, edge_weight=None, edge_attr=None, batch=None, batch_size=None,
                num_sampled_nodes_per_hop=None, num_sampled_edges_per_hop=None, edge_label_index=None, num_graphs=None):
        h = x
        for space_conv in self.encoder.space_convs:
            h = space_conv(h, edge_index, edge_weight)
        output = self.clf(h, edge_label_index, sigmoid=True)
        return output

    def training_step(self, batch, batch_idx):
        h, target_pos_edge_index, target_neg_edge_index = self._process_batch(batch)
        return self._compute_and_log_metrics("train", h, target_pos_edge_index, target_neg_edge_index, self.pos_weight)

    def validation_step(self, batch, batch_idx):
        h, target_pos_edge_index, target_neg_edge_index = self._process_batch(batch)
        return self._compute_and_log_metrics("val", h, target_pos_edge_index, target_neg_edge_index, self.pos_weight)

    def test_step(self, batch, batch_idx):
        h, target_pos_edge_index, target_neg_edge_index = self._process_batch(batch)
        return self._compute_and_log_metrics("test", h, target_pos_edge_index, target_neg_edge_index)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.clf.parameters()), lr=1e-3)#, weight_decay=0.01)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval":"epoch", "frequency":1}
        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler_config}
        #return optimizer









