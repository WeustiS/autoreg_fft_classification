p_s = (4,3)
i_s = (64, 33)
CONFIG = {
    "batch_size": 128,
    "num_workers": 4,
    'hidden_dim': 192,
    'depth': 6,
    'heads': 26,
    "epochs": 200, 
    "wandb": True,
    "frac_tokens": 0.75,
    "n_classes": 200,
    "patch_h": 4,
    "patch_w": 3,
    'weight_decay': .1,
    'randaug_n-op': 2,
    'randaug_mag': 15,
    'dropout': .2
} 
