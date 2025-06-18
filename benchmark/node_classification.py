import torch
import torch_geometric as pyg
from sklearn.metrics import accuracy_score
from torch_geometric.data.lightning import LightningNodeData
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from gnn_lightning import NC_GNN

default_hyperparameters = {
    'in_channels': -1,
    'hidden_channels': 20,
    'num_layers': 3,
    'dropout': 0.2,
    'optimizer_constructor': torch.optim.Adam,
    'lr': 0.001,
    'criterion': torch.nn.CrossEntropyLoss(),
    'loader': 'neighbor',
    'batch_size': 1,
    'num_epochs': 5,
    'enable_progress_bar': True,
    'std': '',
}


def node_classification(model_name, nc_datasets, metrics_save_path, model_save_path, **kwargs):
    default_hyperparameters.update(kwargs)
    if model_name == 'graphsage':
        default_hyperparameters['batch_size'] = 4
        return nc_graphsage(nc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gcn':
        default_hyperparameters['loader'] = 'full'
        return nc_gcn(nc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gat':
        default_hyperparameters['loader'] = 'full'
        return nc_gat(nc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    elif model_name == 'gin':
        default_hyperparameters['loader'] = 'full'
        return nc_gin(nc_datasets, metrics_save_path, model_save_path, default_hyperparameters)
    else:
        raise ValueError(f'Unknown model name: {model_name}')


def nc_graphsage(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'graphsage'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GraphSAGE
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']

    return nc_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                               optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                               metrics_save_path, model_save_path, model_name, enable_progress_bar)
    # return nc_train_traditional(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
    #                             optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
    #                             metrics_save_path, model_save_path, model_name, enable_progress_bar)


def nc_gcn(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gcn'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GCN
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']

    return nc_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                               optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                               metrics_save_path, model_save_path, model_name, enable_progress_bar)
    # return nc_train_traditional(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
    #                             optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
    #                             metrics_save_path, model_save_path, model_name, enable_progress_bar)


def nc_gat(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gat'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GAT
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']

    return nc_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                               optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                               metrics_save_path, model_save_path, model_name, enable_progress_bar)
    # return nc_train_traditional(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
    #                             optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
    #                             metrics_save_path, model_save_path, model_name, enable_progress_bar)


def nc_gin(nc_datasets, metrics_save_path, model_save_path, hyperparameters):
    model_name = 'gin'
    in_channels = hyperparameters['in_channels']
    hidden_channels = hyperparameters['hidden_channels']
    num_layers = hyperparameters['num_layers']
    dropout = hyperparameters['dropout']
    model_constructor = pyg.nn.GIN
    optimizer_constructor = hyperparameters['optimizer_constructor']
    lr = hyperparameters['lr']
    criterion = hyperparameters['criterion']
    loader = hyperparameters['loader']
    batch_size = hyperparameters['batch_size']
    num_epochs = hyperparameters['num_epochs']
    enable_progress_bar = hyperparameters['enable_progress_bar']

    return nc_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                              optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
                              metrics_save_path, model_save_path, model_name, enable_progress_bar)
    #return nc_train_traditional(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
    #                             optimizer_constructor, lr, criterion, loader, num_epochs, batch_size,
    #                             metrics_save_path, model_save_path, model_name, enable_progress_bar)


def nc_train_traditional(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                            optimizer_constructor, lr, criterion, loader, num_epochs, batch_size, metrics_save_path,
                            model_save_path, model_name, enable_progress_bar):
    loggers = []
    for dataset_name, data in nc_datasets:
        print('$' * 101)
        print(f'Training on {dataset_name} dataset')
        out_channels = data.num_classes
        is_multiclass = data.num_classes > 2
        if data.num_classes == 2:
            out_channels = 1
            criterion = torch.nn.BCEWithLogitsLoss()

        model = model_constructor(in_channels=in_channels, hidden_channels=hidden_channels,
                                  out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        print(model)
        optimizer = optimizer_constructor(model.parameters(), lr=lr)
        model.train()
        train_loss = train_acc = val_loss = val_acc = test_loss = test_acc = 0
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            train_loss = criterion(out[data.train_mask], data.y[data.train_mask])
            y_pred = out.argmax(dim=1)

            train_acc = accuracy_score(y_pred[data.train_mask].cpu().numpy(), data.y[data.train_mask].cpu().numpy())

            val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
            val_acc = (out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()


            train_loss.backward()
            optimizer.step()

            if epoch % 5 == 0:
                print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        model.eval()
        out = model(data.x, data.edge_index)
        test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
        test_acc = (out[data.test_mask].argmax(dim=1) == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    return loggers


def nc_train_lightning(nc_datasets, model_constructor, in_channels, hidden_channels, num_layers, dropout,
                       optimizer_constructor, lr, criterion, loader, num_epochs, batch_size, metrics_save_path,
                       model_save_path, model_name, enable_progress_bar):
    loggers = []
    std = default_hyperparameters['std']
    for dataset_name, data in nc_datasets:
        print('$' * 101)
        print(f'Training on {dataset_name} dataset')
        out_channels = data.num_classes
        is_multiclass = data.num_classes > 2
        if data.num_classes == 2:
            out_channels = 1
            criterion = torch.nn.BCEWithLogitsLoss()

        model = model_constructor(in_channels=in_channels, hidden_channels=hidden_channels,
                                  out_channels=out_channels, num_layers=num_layers, dropout=dropout)
        optimizer = optimizer_constructor(model.parameters(), lr=lr)
        nc_gnn = NC_GNN(model, optimizer, criterion, is_multiclass, is_lp=False)
        logger = CSVLogger(metrics_save_path, name=f'nc_{model_name}_{dataset_name}{std}')
        if loader == 'full':
            datamodule = LightningNodeData(
                data,
                input_train_nodes=data.train_mask,
                input_val_nodes=data.val_mask,
                input_test_nodes=data.test_mask,
                loader=loader,
                batch_size=batch_size
            )
        else:
            datamodule = LightningNodeData(
                data,
                input_train_nodes=data.train_mask,
                input_val_nodes=data.val_mask,
                input_test_nodes=data.test_mask,
                loader=loader,
                num_neighbors=[int(data.num_edges / data.num_nodes) for _ in
                               range(num_layers)] if loader == 'neighbor' else None,
                batch_size=batch_size
            )
        trainer = pl.Trainer(max_epochs=num_epochs, logger=logger, enable_progress_bar=enable_progress_bar)
        trainer.fit(nc_gnn, datamodule=datamodule)
        trainer.test(datamodule=datamodule)

        torch.save(nc_gnn, model_save_path + f'nc_{model_name}_{dataset_name}{std}_model.pth')
        loggers.append((dataset_name, logger))
    return loggers