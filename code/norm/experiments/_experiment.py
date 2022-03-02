from ._run import Run
from ._validation import run_valid as validate_run
from ._test import run_test
from norm._typings import BaseLoader
from norm import get_date
from norm.io import dump
from pathlib import Path
from typing import Callable, List


def get_name():
    from randomname import generate
    return generate('v/programming', 'n/algorithms')


def _init_ray(num_cpus: int, num_gpus: int, nw: int):
    import ray
    ray.init(num_cpus=num_cpus,
             num_gpus=num_gpus,
             include_dashboard=False)

    @ray.remote(num_cpus=1, num_gpus=1/nw)
    def run_valid(*args, **kwargs) -> float:
        return validate_run(*args, **kwargs)

    return run_valid


def _run_seq(exp, tr_set, vl_set, save_path):

    LOW_ = 0.0
    HIGH_ = 1e4
    best_score = LOW_ if exp.higher_is_better else HIGH_

    is_better = lambda a, b: a > b if exp.higher_is_better else a < b  # noqa: E731

    for run in exp.runs:
        score = exp._fire(run, exp.evaluate_fn, tr_set, vl_set, save_path / exp.name, exp.num_valid_trials, exp.higher_is_better)
        if is_better(score, best_score):
            best_score, best_run = score, run

    return best_score, best_run


def _run_par(exp, tr_set, vl_set, save_path):
    import ray
    info_ = []

    for run in exp.runs:
        id_ = exp._fire.remote(run, exp.evaluate_fn, tr_set, vl_set, save_path / exp.name, exp.num_valid_trials, exp.higher_is_better)
        info_.append((id_, run))

    LOW_ = 0.0
    HIGH_ = 1e4
    best_score = LOW_ if exp.higher_is_better else HIGH_
    is_better = lambda a, b: a > b if exp.higher_is_better else a < b  # noqa: E731

    for id_, run_ in info_:
        score = ray.get(id_)
        if is_better(score, best_score):
            best_score, best_run = score, run_

    return best_score, best_run


class Experiment:
    def __init__(self,
                 runs: List[Run],
                 evaluate_fn: Callable,
                 save_path: Path,
                 num_cpus: int = 1,
                 num_gpus: int = 0,
                 num_valid_trials: int = 1,
                 num_test_trials: int = 5,
                 nw: int = 1,
                 higher_is_better: bool = True):
        self.evaluate_fn = evaluate_fn
        self.runs = runs
        self.sequential = num_cpus == 1
        self.higher_is_better = higher_is_better
        self.num_valid_trials = num_valid_trials
        self.num_test_trials = num_test_trials

        if not self.sequential:
            self._fire = _init_ray(num_cpus, num_gpus, nw)
        else:
            self._fire = validate_run

        # make sure to generate a unique id for the experiment (try up to 10 possible names)
        for _ in range(10):
            self.name = get_name()
            if not (save_path / self.name).exists():
                break

        assert not (save_path / self.name).exists(), f"Experiment '{self.name}' already existing."


def validate(experiment: Experiment,
             tr_set: BaseLoader,
             vl_set: BaseLoader,
             save_path: Path) -> Run:

    if len(experiment.runs) == 1:
        best_run = experiment.runs[0]
        dump({
            'meta': {
                'name': best_run.name,
                'vl_score': 'ND',
                'date': get_date()
            },
            'config': best_run.config,
        }, save_path / experiment.name / 'best_run.json')
        return best_run

    if experiment.sequential:
        best_score, best_run = _run_seq(experiment,
                                        tr_set,
                                        vl_set,
                                        save_path)
    else:
        best_score, best_run = _run_par(experiment,
                                        tr_set,
                                        vl_set,
                                        save_path)

    dump({
        'meta': {
            'name': best_run.name,
            'vl_score': best_score,
            'date': get_date()
        },
        'config': best_run.config,
    }, save_path / experiment.name / 'best_run.json')

    return best_run


def run_exp(experiment: Experiment,
            tr_set: BaseLoader,
            vl_set: BaseLoader,
            ts_set: BaseLoader,
            save_path: Path):

    best_run = validate(experiment=experiment,
                        tr_set=tr_set,
                        vl_set=vl_set,
                        save_path=save_path)

    run_test(run=best_run,
             evaluate_fn=experiment.evaluate_fn,
             tr_set=tr_set + vl_set,
             ts_set=ts_set,
             num_trials=experiment.num_test_trials,
             higher_is_better=experiment.higher_is_better,
             save_path=save_path / experiment.name)