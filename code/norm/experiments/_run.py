from ._random_search import random_search
from clrs import Model
from dataclasses import dataclass
from functools import partial
from torch.optim import Optimizer
from typing import Any, Callable, Dict


@dataclass(init=True, eq=True, frozen=True)
class Run:
    name: str
    batch_size: int
    model_fn: Callable[[Any], Model]
    config: Dict[str, Any]
    early_stop: bool
    early_stop_patience: int
    log_every: int
    train_steps: int
    verbose: bool


def init_runs(num_runs: int,
              hp_space: Dict,
              model_fn: Callable[[Any], Model],
              optim_fn: Callable[[Any], Optimizer],
              batch_size: int,
              early_stop: bool,
              early_stop_patience: int,
              log_every: int,
              train_steps: int,
              verbose: bool):

    assert 'model' in hp_space, "Model's hyperparameters are missing"
    assert 'optim' in hp_space, "Optimiser's hyperparameters are missing"

    model_configs = hp_space['model']
    optim_configs = hp_space['optim']

    search = partial(random_search, num_samples=num_runs)
    runs = list()

    for i, (m_conf, o_conf) in enumerate(zip(search(model_configs), search(optim_configs))):

        model_fn = partial(model_fn,
                           **m_conf,
                           optim_fn=partial(optim_fn, **o_conf))

        run_ = Run(name=f'run_{i}',
                   config={**m_conf, **o_conf},
                   model_fn=model_fn,
                   batch_size=batch_size,
                   early_stop=early_stop,
                   early_stop_patience=early_stop_patience,
                   log_every=log_every,
                   train_steps=train_steps,
                   verbose=verbose)

        runs.append(run_)

    return runs
