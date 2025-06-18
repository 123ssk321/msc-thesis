import torch
import torch_geometric as pyg
import torch_geometric.nn
from torch_geometric.loader import DataLoader, LinkNeighborLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from gnn_lightning import LP_GNN

default_hyperparameters = {
    'in_channels': -1,
    'hidden_channels': 20,
    'num_layers': 2,
    'dropout': 0.2,
    'decoder': torch_geometric.nn.InnerProductDecoder(),
    'optimizer_constructor': torch.optim.Adam,
    'lr': 0.01,
    'criterion': torch.nn.BCEWithLogitsLoss(),
    'loader': 'neighbor',
    'batch_size': 4,
    'num_epochs': 5,
    'enable_progress_bar': True,
    'std': '',
}


def link_prediction(model_name, lp_datasets, metrics_save_path, model_save_path, **kwargs):
    default_hyperparameters.update(kwargs)
    if model_name == 'graphsage':
        default_hyperparameters['batch_size'] = 4
        return lp_graphsage(lp_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gcn':
        default_hyperparameters['loader'] = 'full'
        return lp_gcn(lp_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gat':
        default_hyperparameters['loader'] = 'full'
        return lp_gat(lp_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gin':
        default_hyperparameters['loader'] = 'full'
        return lp_gin(lp_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def lp_graphsage(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'graphsage'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GraphSAGE
    decoder = hyperparameters['decoder']
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return lp_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                              decoder,
                              optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def lp_gcn(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gcn'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GCN
    decoder = hyperparameters['decoder']
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return lp_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout, decoder,
                              optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def lp_gat(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gat'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GAT
    decoder = hyperparameters['decoder']
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return lp_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout, decoder,
                              optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def lp_gin(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gin'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GIN
    decoder = hyperparameters['decoder']
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']
    std = hyperparameters['std']

    return lp_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout, decoder,
                              optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar, std)


def lp_train_lightning(lp_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout, decoder,
                       optimizer_constructor, lr, criterion, loader, num_epochs, batch_size, metrics_save_path,
                       model_save_path, model_name, enable_progress_bar, std):
    loggers = []
    for dataset_name, train_data, val_data, test_data in lp_datasets:
        print('$' * 101)
        print(f'Training on {dataset_name} dataset')

        model = model_constructor(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout)
        optimizer = optimizer_constructor(model.parameters(), lr=lr)
        lp_gnn = LP_GNN(model, optimizer, criterion, False, True, decoder)
        logger = CSVLogger(metrics_save_path, name=f'lp_{model_name}_{dataset_name}{std}')
        if loader == 'full':
            train_loader = DataLoader([train_data], batch_size=batch_size, shuffle=True)
            val_loader = DataLoader([val_data], batch_size=batch_size, shuffle=False)
            test_loader = DataLoader([test_data], batch_size=batch_size, shuffle=False)
        else:
            num_samples = [20 for _ in range(num_layers)]
            train_loader = LinkNeighborLoader(
                data=train_data,
                num_neighbors=num_samples,
                edge_label_index=train_data.edge_label_index,
                edge_label=train_data.edge_label,
                batch_size=batch_size,
                shuffle=True,
                subgraph_type='bidirectional' if train_data.is_undirected() else 'directional'
            )

            val_loader = LinkNeighborLoader(
                data=val_data,
                num_neighbors=num_samples,
                edge_label_index=val_data.edge_label_index,
                edge_label=val_data.edge_label,
                batch_size=batch_size,
                shuffle=True,
                subgraph_type='bidirectional' if val_data.is_undirected() else 'directional'
            )

            test_loader = LinkNeighborLoader(
                data=test_data,
                num_neighbors=num_samples,
                edge_label_index=test_data.edge_label_index,
                edge_label=test_data.edge_label,
                batch_size=batch_size,
                shuffle=True,
                subgraph_type='bidirectional' if test_data.is_undirected() else 'directional'
            )

        trainer = pl.Trainer(max_epochs=num_epochs, logger=logger, enable_progress_bar=enable_progress_bar)
        trainer.fit(lp_gnn, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(dataloaders=test_loader)

        torch.save(lp_gnn, model_save_path + f'lp_{model_name}_{dataset_name}{std}_model.pth')
        loggers.append((dataset_name, logger))
    return loggers
