sweep_config = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'val_accuracy',
    },
    'parameters': {
        'optimizer': {
            'values': ['Adam', 'AdamW', 'SGD'],
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5,
        },
        'lr': {
            'distribution': 'uniform',
            'min': 1e-5,
            'max': 0.5,
        },
        'epochs': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 50,
        },
        'batch_size_exp': {
            'distribution': 'int_uniform',
            'min': 3,
            'max': 10,
        },
        'k': {
            'distribution': 'uniform',
            'min': 0.05,
            'max': 0.65,
        }
    }
}