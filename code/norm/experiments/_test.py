from ._run import Run
from norm._typings import BaseLoader
from norm.io import dump
from typing import Callable
from pathlib import Path
from rich.pretty import pprint as log
from statistics import mean, stdev


def run_test(run: Run,
             evaluate_fn: Callable,
             tr_set: BaseLoader,
             ts_set: BaseLoader,
             save_path: Path,
             num_trials: int = 5,
             higher_is_better: bool = True):

    is_better = lambda a, b: a > b if higher_is_better else a < b  # noqa: E731
    is_best = lambda a, b: a > max(b) if higher_is_better else a < min(b)  # noqa: E731

    LOW_ = 0.0
    HIGH_ = 1e4

    def closure():
        model = run.model_fn()
        best = LOW_ if higher_is_better else HIGH_
        losses, tr_scores, ts_scores = [], [], []

        for step in range(1, run.train_steps+1):
            feedback = tr_set.next(run.batch_size)
            loss = model.feedback(feedback)
            losses.append(loss)

            if step % run.log_every == 0:
                tr_stats = evaluate_fn(model, feedback, extras={'step': step, 'loss': loss})
                ts_stats = evaluate_fn(model, ts_set.next(), extras={'step': step}, verbose=run.verbose)

                tr_scores.append(tr_stats)
                ts_scores.append(ts_stats)

                log(dict(
                    **{
                        f'tr_{key}': item
                        for key, item in tr_stats.items()
                    },
                    **{
                        f'ts_{key}': item
                        for key, item in ts_stats.items()
                    }
                ))

                if is_better(ts_stats['score'], best):
                    best = ts_stats['score']
                    best_score[trial] = best
                    model.dump_model(save_path / f'model_{trial}.pth')

                if is_best(best, best_score):
                    dump(dict(
                        trial=trial+1,
                        **ts_stats,
                    ), save_path / 'test_stats.json')
                    model.dump_model(save_path / 'model.pth')

        return losses, tr_scores, ts_scores

    losses, tr_scores, ts_scores, best_score = [], [], [], [LOW_ if higher_is_better else HIGH_]*5

    for trial in range(num_trials):
        loss, tr_score, ts_score = closure()
        losses.append(loss)
        tr_scores.append(tr_score)
        ts_scores.append(ts_score)

    dump(dict(
        loss=losses,
        num_trials=num_trials,
        tr_scores=tr_scores,
        ts_scores=ts_scores,
        mean_score=mean(best_score),
        std_score=stdev(best_score) if num_trials > 1 else 0
    ), save_path / 'test_info.json')
