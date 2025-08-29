import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from ultralytics.eval import compute_map

RUN_NAME_REGEX = re.compile(r'([a-z0-9.-]*)\.json_([a-z-0-9]*)_kfold_([0-9])\.yaml')


def average_dict(dicts: List[Dict]):
    num = len(dicts)

    sum = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            sum[k] += v

    for k in sum.keys():
        sum[k] /= num

    return sum


if __name__ == '__main__':
    logging.disable()

    root = Path('/datasets/yolo_runs')
    all_folds = sorted(f for f in root.iterdir() if f.is_dir())[::-1]

    grouped_by_model_folds = defaultdict(list)
    for fold in all_folds:
        res = re.search(RUN_NAME_REGEX, fold.name)
        model, split, kfold = res[1], res[2], res[3]
        grouped_by_model_folds[(model, split)].append(fold)

    for (model, split), paths in grouped_by_model_folds.items():
        folds = [compute_map(path)['map'] for path in paths]
        avg_metric = average_dict(folds)
        print('-' * 20)
        print(model, split)
        print(avg_metric)
