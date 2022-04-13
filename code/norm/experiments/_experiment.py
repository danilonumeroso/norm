from ._run import Run
from ._validation import run_valid as validate_run
from ._test import run_test
from norm._typings import BaseLoader
from norm import get_date
from norm.io import dump, load
from pathlib import Path
from typing import Callable, List

NOT_DEFINED = "ND"
RESUME_EXP_DIR = "__experiment_status__"


def get_name():
    from randomname import generate
    return generate('v/programming', 'n/algorithms')


def _init_ray(num_cpus: int, num_gpus: int, nw: int):
    import ray
    ray.init(num_cpus=num_cpus,
             num_gpus=num_gpus,
             include_dashboard=False)


def _run_seq(exp, tr_set, vl_set, save_path):

    HIGH_ = 1e4
    LOW_ = -HIGH_
    best_score = LOW_ if exp.higher_is_better else HIGH_

    is_better = lambda a, b: a > b if exp.higher_is_better else a < b  # noqa: E731

    for run in exp.runs:
        score = exp._fire(run, exp.evaluate_fn, tr_set, vl_set, save_path / exp.name, exp.num_valid_trials, exp.higher_is_better)
        if is_better(score, best_score):
            best_score, best_run = score, run

    return best_score, best_run


def _run_par(exp, tr_set, vl_set, save_path):
    import ray
    remotes = {}

    for run in exp.runs:
        id_ = exp._fire.remote(run, exp.evaluate_fn, tr_set, vl_set, save_path / exp.name, exp.num_valid_trials, exp.higher_is_better)
        remotes[id_] = run

    HIGH_ = 1e4
    LOW_ = -HIGH_

    best_score = LOW_ if exp.higher_is_better else HIGH_
    is_better = lambda a, b: a > b if exp.higher_is_better else a < b  # noqa: E731

    try:
        for id_, run_ in remotes.items():
            score = ray.get(id_)
            if is_better(score, best_score):
                best_score, best_run = score, run_
    except KeyboardInterrupt:
        print('Received a Keyboard Interrupt, saving current state...')
        completed, uncompleted = ray.wait(list(remotes.keys()), timeout=1)
        _save_and_interrupt(exp, remotes, completed, uncompleted, save_path)
        raise KeyboardInterrupt("Experiment interrupted.")

    return best_score, best_run


def _save_and_interrupt(exp, remotes, completed_runs, uncompleted_runs, save_path):
    import ray

    for id in uncompleted_runs:
        ray.cancel(id, force=True)

    r_status = {}
    r_scores = {}

    for id in remotes:
        run_name = remotes[id].name
        r_status[run_name] = id in completed_runs
        r_scores[run_name] = ray.get(id) if id in completed_runs else NOT_DEFINED

    dump(r_status, save_path / exp.name / RESUME_EXP_DIR / 'runs.status.json')
    dump(r_scores, save_path / exp.name / RESUME_EXP_DIR / 'runs.scores.json')

    exp._fire = None
    dump(exp, save_path / exp.name / RESUME_EXP_DIR / 'experiment.status.pkl')


class Experiment:
    def __init__(self,
                 runs: List[Run],
                 evaluate_fn: Callable,
                 save_path: Path,
                 name: str = '',
                 num_cpus: int = 1,
                 num_gpus: int = 0,
                 num_valid_trials: int = 1,
                 num_test_trials: int = 5,
                 nw: int = 1,
                 higher_is_better: bool = True):

        self.runs = runs
        self.evaluate_fn = evaluate_fn
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.num_valid_trials = num_valid_trials
        self.num_test_trials = num_test_trials
        self.nw = nw
        self.higher_is_better = higher_is_better

        self.sequential = num_cpus == 1
        if not self.sequential:
            _init_ray(num_cpus, num_gpus, nw)

            from ray import remote

            @remote(num_cpus=1, num_gpus=1/nw if num_gpus > 0 else 0)
            def run_valid(*args, **kwargs) -> float:
                return validate_run(*args, **kwargs)

            self._fire = run_valid

        else:
            self._fire = validate_run

        self.name = name + '-' + get_name()

        if (save_path / self.name).exists():
            print(f"[warning]: experiment {self.name} already existing.")
            self.name += "-new"


def validate(experiment: Experiment,
             tr_set: BaseLoader,
             vl_set: BaseLoader,
             save_path: Path) -> Run:

    if len(experiment.runs) == 1:
        best_run = experiment.runs[0]
        dump({
            'meta': {
                'name': best_run.name,
                'vl_score': NOT_DEFINED,
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
             tr_set=tr_set,
             vl_set=vl_set,
             ts_set=ts_set,
             num_trials=experiment.num_test_trials,
             higher_is_better=experiment.higher_is_better,
             save_path=save_path / experiment.name)


def resume_exp(experiment_path: Path) -> Experiment:
    r_status = load(experiment_path / RESUME_EXP_DIR / 'runs.status.json')
    r_scores = load(experiment_path / RESUME_EXP_DIR / 'runs.scores.json')
    exp = load(experiment_path / RESUME_EXP_DIR / 'experiment.status.pkl')

    if not exp.sequential:
        _init_ray(exp.num_cpus, exp.num_gpus, exp.nw)

        from ray import remote

        @remote(num_cpus=1, num_gpus=1/exp.nw if exp.num_gpus > 0 else 0)
        def run_valid(run, *args, **kwargs) -> float:
            if r_status[run.name]:
                return r_scores[run.name]

            return validate_run(run, *args, **kwargs)

        exp._fire = run_valid
    else:
        raise NotImplementedError("Resume is not yet implemented for sequential experiments.")
        exp._fire = validate_run

    from shutil import rmtree
    rmtree(experiment_path / RESUME_EXP_DIR)

    return exp
