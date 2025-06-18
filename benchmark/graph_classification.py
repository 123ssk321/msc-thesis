import random

import torch
import torch_geometric as pyg
from torch_geometric.data.lightning import LightningDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from gnn_lightning import GC_GNN

default_hyperparameters = {
    'in_channels': -1,
    'hidden_channels': 20,
    'num_layers': 3,
    'dropout': 0.2,
    'pooling_type': 'max',
    'optimizer_constructor': torch.optim.Adam,
    'lr': 0.001,
    'criterion': torch.nn.CrossEntropyLoss(),
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'batch_size': 1,
    'num_epochs': 5,
    'enable_progress_bar': True,
    'std': '',
}


def graph_classification(model_name, gc_datasets, metrics_save_path, model_save_path, **kwargs):
    default_hyperparameters.update(kwargs)
    if model_name == 'graphsage':
        return gc_graphsage(gc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gcn':
        return gc_gcn(gc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gat':
        return gc_gat(gc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gin':
        return gc_gin(gc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def gc_graphsage(gc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'graphsage'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    pooling_type = hyperparameters['pooling_type']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GraphSAGE
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    train_ratio = hyperparameters['train_ratio']
    val_ratio = hyperparameters['val_ratio']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return gc_train_lightning(gc_datasets, train_ratio, val_ratio, model_constructor, in_channels, hidden_channels,
                              num_layers, pooling_type, dropout, optimizer_constructor, lr, criterion, num_epochs,
                              batch_size, metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def gc_gcn(gc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gcn'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    pooling_type = hyperparameters['pooling_type']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GCN
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    train_ratio = hyperparameters['train_ratio']
    val_ratio = hyperparameters['val_ratio']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return gc_train_lightning(gc_datasets, train_ratio, val_ratio, model_constructor, in_channels, hidden_channels,
                              num_layers,
                              pooling_type, dropout, optimizer_constructor, lr, criterion, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def gc_gat(gc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gat'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    pooling_type = hyperparameters['pooling_type']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GAT
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    train_ratio = hyperparameters['train_ratio']
    val_ratio = hyperparameters['val_ratio']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return gc_train_lightning(gc_datasets, train_ratio, val_ratio, model_constructor, in_channels, hidden_channels,
                              num_layers,
                              pooling_type, dropout, optimizer_constructor, lr, criterion, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def gc_gin(gc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gin'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    pooling_type = hyperparameters['pooling_type']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GIN
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    train_ratio = hyperparameters['train_ratio']
    val_ratio = hyperparameters['val_ratio']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return gc_train_lightning(gc_datasets, train_ratio, val_ratio, model_constructor, in_channels, hidden_channels,
                              num_layers,
                              pooling_type, dropout, optimizer_constructor, lr, criterion, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def gc_train_lightning(gc_datasets, train_ratio, val_ratio, model_constructor, in_channels, hidden_channels, num_layers,
                       pooling_type, dropout,
                       optimizer_constructor, lr, criterion, num_epochs, batch_size,
                       metrics_save_path, model_save_path, model_name, enable_progress_bar, std):
    loggers = []
    for dataset_name, data_list, dataset_num_classes in gc_datasets:
        print('$' * 101)
        print(f'Training on {dataset_name} dataset')
        model = model_constructor(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout)
        optimizer = optimizer_constructor(model.parameters(), lr=lr)
        is_multiclass = dataset_num_classes > 2
        if dataset_num_classes == 2:
            dataset_num_classes = 1
            criterion = torch.nn.BCEWithLogitsLoss()
        gc_gnn = GC_GNN(model, optimizer, criterion, is_multiclass, False, dataset_num_classes, pooling_type)
        logger = CSVLogger(metrics_save_path, name=f'gc_{model_name}_{dataset_name}{std}')

        #random.shuffle(data_list)
        train_size = int(train_ratio * len(data_list))
        val_size = int(val_ratio * len(data_list))
        train_dataset = data_list[:train_size]
        val_dataset = data_list[train_size:train_size + val_size]
        test_dataset = data_list[train_size + val_size:]
        datamodule = LightningDataset(train_dataset, val_dataset, test_dataset,
                                      batch_size=batch_size, shuffle=True)

        trainer = pl.Trainer(max_epochs=num_epochs, logger=logger, enable_progress_bar=enable_progress_bar)
        trainer.fit(gc_gnn, datamodule=datamodule)
        trainer.test(datamodule=datamodule)

        torch.save(gc_gnn, model_save_path + f'gc_{model_name}_{dataset_name}{std}_model.pth')
        loggers.append((dataset_name, logger))
    return loggers
