from ._run import Run
from norm._typings import BaseLoader
from norm.io import dump
from typing import Callable
from pathlib import Path
from rich.pretty import pprint as log


def run_test(run: Run,
             evaluate_fn: Callable,
             tr_set: BaseLoader,
             vl_set: BaseLoader,
             ts_set: BaseLoader,
             save_path: Path,
             num_trials: int = 5,
             name: str = '',
             higher_is_better: bool = True):

    is_better = lambda a, b: a > b if higher_is_better else a < b  # noqa: E731

    LOW_ = 0.0
    HIGH_ = 1e4

    def closure():
        model = run.model_fn()
        best = LOW_ if higher_is_better else HIGH_
        losses, tr_scores, vl_scores = [], [], []

        for step in range(1, run.train_steps+1):
            feedback = tr_set.next(run.batch_size)
            loss = model.feedback(feedback)
            losses.append(loss)

            if step % run.log_every == 0:
                tr_stats = evaluate_fn(model, feedback, extras={'step': step, 'loss': loss})
                vl_stats = evaluate_fn(model, vl_set.next(), extras={'step': step}, verbose=run.verbose)

                tr_scores.append(tr_stats)
                vl_scores.append(vl_stats)

                log(dict(
                    **{
                        f'tr_{key}': item
                        for key, item in tr_stats.items()
                    },
                    **{
                        f'vl_{key}': item
                        for key, item in vl_stats.items()
                    }
                ))

                if is_better(vl_stats['score'], best):
                    best = vl_stats['score']
                    dump(model.net_.state_dict(), save_path / f'model_{trial}.pth')

        return losses, tr_scores, vl_scores

    losses, tr_scores, vl_scores, ts_stats, ts_scores = [], [], [], [], []

    for trial in range(num_trials):
        loss, tr_score, vl_score = closure()
        losses.append(loss)
        tr_scores.append(tr_score)
        vl_scores.append(vl_score)

        model = run.model_fn()
        model.restore_model(save_path / f'model_{trial}.pth', 'cpu')
        ts_stats.append(evaluate_fn(model, ts_set.next()))
        ts_scores.append(ts_stats[-1]['score'])

    return dict(
        loss=losses,
        num_trials=num_trials,
        tr_scores=tr_scores,
        vl_scores=vl_scores,
        ts_scores=ts_scores,
        ts_stats=ts_stats)
