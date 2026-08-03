"""One place to seed everything. Call set_all_seeds() first in every script."""
import os
import random

import numpy as np


def set_all_seeds(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def make_sampler(seed: int):
    """Seeded Optuna sampler. HPO is not reproducible without this."""
    import optuna

    return optuna.samplers.TPESampler(seed=seed)


def worker_init_fn(worker_id: int):
    """Pass to DataLoader(worker_init_fn=...) so augmentation is reproducible."""
    seed = (int(os.environ.get("PYTHONHASHSEED", 0)) + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
