import torch

from data_store import nc_datasets, gc_datasets, lp_datasets

model_names = ['gcn', 'graphsage', 'gat', 'gin']
stds = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']


def get_gnn(model_path, task, model_name, dataset_name, std=None):
    if task == 'nc':
        datasets = nc_datasets
    elif task == 'gc':
        datasets = gc_datasets
    elif task == 'lp':
        datasets = lp_datasets
    else:
        raise ValueError(f'Invalid task: {task}')
    model_path = f'{model_path}{dataset_name}/'
    if model_name in model_names:
        if dataset_name in datasets:
            if std == 'all':
                models = []
                for std in stds:
                    path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                    model = torch.load(path)
                    models.append((std, model))
                return models
            elif std in stds:
                path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                model = torch.load(path)
                return model
            path = f'{model_path}{task}_{model_name}_{dataset_name}_model.pth'
            model = torch.load(path)
            return model
        if dataset_name == 'all':
            models = []
            if std == 'all':
                for std in stds:
                    for dataset_name in datasets:
                        path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                        model = torch.load(path)
                        models.append((dataset_name, std, model))
                return models
            elif std in stds:
                for dataset_name in datasets:
                    path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                    model = torch.load(path)
                    models.append((dataset_name, model))
                return models
            for dataset_name in datasets:
                path = f'{model_path}{task}_{model_name}_{dataset_name}_model.pth'
                model = torch.load(path)
                models.append((dataset_name, model))
            return models
    if model_name == 'all':
        models = []
        if dataset_name in datasets:
            if std == 'all':
                for std in stds:
                    for model_name in model_names:
                        path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                        model = torch.load(path)
                        models.append((model_name, std, model))
                return models
            elif std in stds:
                for model_name in model_names:
                    path = f'{model_path}{task}_{model_name}_{dataset_name}{std}_model.pth'
                    model = torch.load(path)
                    models.append((model_name, model))
                return models
            for model_name in model_names:
                path = f'{model_path}{task}_{model_name}_{dataset_name}_model.pth'
                model = torch.load(path)
                models.append((model_name, model))
            return models
        if dataset_name == 'all':
            for model_name in model_names:
                for dataset_name in datasets:
                    path = f'{model_path}{task}_{model_name}_{dataset_name}_model.pth'
                    model = torch.load(path)
                    models.append((model_name, dataset_name, model))
            return models
        return models
