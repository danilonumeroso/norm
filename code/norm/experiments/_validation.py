from ._run import Run
from norm._typings import BaseLoader
from norm import set_seed
from norm.io import dump
from pathlib import Path
from rich.pretty import pprint as log
from statistics import mean, stdev
from typing import Callable


def _early_stop(loss, patience):
    from math import isnan
    return isnan(loss) or patience <= 0


def run_valid(run: Run,
              evaluate_fn: Callable,
              tr_set: BaseLoader,
              vl_set: BaseLoader,
              save_path: Path,
              num_trials: int = 1,
              higher_is_better: bool = True) -> float:

    set_seed(run.seed)
    is_better = lambda a, b: a > b if higher_is_better else a < b  # noqa: E731
    log(run.config)

    HIGH_ = 1e4
    LOW_ = -HIGH_

    def closure():
        model = run.model_fn()
        best_score = LOW_ if higher_is_better else HIGH_
        patience_ = run.early_stop_patience
        losses, tr_scores, vl_scores = [], [], []

        for step in range(1, run.train_steps + 1):
            feedback = tr_set.next(run.batch_size)
            loss = model.feedback(feedback)
            losses.append(loss)

            if step % run.log_every == 0:
                tr_stats = evaluate_fn(model, feedback, extras={'step': step, 'loss': loss}, verbose=run.verbose)
                vl_stats = evaluate_fn(model, vl_set.next(), extras={'step': step}, verbose=run.verbose)

                tr_scores.append(tr_stats)
                vl_scores.append(vl_stats)

                log(dict(
                    run_id=run.name,
                    **{
                        f'tr_{key}': item
                        for key, item in tr_stats.items()
                    },
                    **{
                        f'vl_{key}': item
                        for key, item in vl_stats.items()
                    }
                ))

                if is_better(vl_stats['score'], best_score):
                    best_score = vl_stats['score']
                else:
                    patience_ = (patience_ - 1) if run.early_stop else patience_

                if run.early_stop and _early_stop(tr_stats['loss'], patience_):
                    print("early stopping...")
                    break

        return losses, tr_scores, vl_scores, best_score

    losses, tr_scores, vl_scores, best_score = [], [], [], []

    for _ in range(num_trials):
        loss, tr_score, vl_score, score = closure()
        losses.append(loss)
        tr_scores.append(tr_score)
        vl_scores.append(vl_score)
        best_score.append(score)

    dump(dict(
        name=run.name,
        seed=run.seed,
        loss=losses,
        num_trials=num_trials,
        tr_scores=tr_scores,
        vl_scores=vl_scores,
        mean_score=mean(best_score),
        std_score=stdev(best_score) if num_trials > 1 else 0,
    ), save_path / run.name / 'train_info.json')

    dump(run.config, save_path / run.name / 'parameters.json')

    return mean(best_score)
