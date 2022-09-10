import wandb
from modules.loops.main_loop import loop

def hparams_tuning(sweep_config, loss_fn, train_ds, *, val_ds=None, project_name=None, in_features=1, out_features=1,save_dir='', device='cpu'):

    sweep_id = wandb.sweep(sweep_config, project=f'{project_name}-sweep')
    wandb.agent(sweep_id, lambda: loop(loss_fn, train_ds, val_ds, project_name, in_features, out_features, save_dir=save_dir, device=device), count=35)