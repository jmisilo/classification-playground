import os
import wandb
import torch
import torch.optim as optim

from .training_loop import train_epoch
from .validation_loop import valid_epoch
from modules.model import Net
from modules.utils.lr_warmup import LRWarmup

def loop(loss_fn, train_ds, val_ds=None, project_name=None, in_features=1, out_features=1, *, save_dir='', config=None, device='cpu'):
    run = wandb.init(config=config, project=project_name, reinit=True)
    with run:
        criterion = loss_fn()
        
        config = wandb.config
        
        batch_size = 2 ** config.batch_size_exp

        is_cuda = not (device == 'cpu' or device.type == 'cpu')

        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2 if is_cuda else 0, 
            pin_memory=is_cuda
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds, 
            batch_size=batch_size, 
            num_workers=2 if is_cuda else 0, 
            pin_memory=is_cuda
        ) if val_ds is not None else None

        model = Net(in_features=in_features, out_features=out_features, p=config.dropout).to(device)
        wandb.watch(model, criterion, log='all', log_freq=(len(train_ds) // (2 * batch_size)))

        if config.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr)

        elif config.optimizer == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=config.lr)

        elif config.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config.lr)

        else:
            raise ValueError(f'Unknown optimizer: {config.optimizer}')
    
        warmup = LRWarmup(epochs=config.epochs, max_lr=config.lr, k=config.k)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup.lr_warmup)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(config.epochs):
            
            train_res = train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, device)
            valid_res = valid_epoch(model, val_loader, criterion, device)

            wandb.log({
                't_loss': train_res['loss'],
                't_accuracy': train_res['accuracy'],
                'val_loss': valid_res['loss'],
                'val_accuracy': valid_res['accuracy'],
                'lr': scheduler.get_last_lr()[0]
            })

            scheduler.step()

        torch.save(model.state_dict(), os.path.join('weights', save_dir, f'{run.name}.pt'))