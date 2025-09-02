import logging
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from ultralytics.eval import compute_map, compute_speed
from ultralytics.utils.metrics import SegmentMetrics
from faster_coco_eval import COCO, COCOeval_faster

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

    root = Path('./yolo_runs4')
    all_folds = sorted(f for f in root.iterdir() if f.is_dir())[::-1]
    output_path = Path('results/')

    grouped_by_model_folds = defaultdict(list)
    for fold in all_folds:
        res = re.search(RUN_NAME_REGEX, fold.name)
        model, split, kfold = res[1], res[2], res[3]
        grouped_by_model_folds[(model, split)].append(fold)

    for i, ((model, split), paths) in enumerate(grouped_by_model_folds.items()):
        print(f'Evaluating {model} on {split}...')
        speeds = [compute_speed(path) for path in sorted(paths)]

        for i, (time_s, n_img) in enumerate(speeds):
            out = output_path / model
            out.mkdir(parents=True, exist_ok=True)
            obj = dict(time_s=time_s, n_img=n_img)
            with open(out / f'split_{i}_speed.json', 'w') as f:
                json.dump(obj, f)
        continue

        folds = [compute_map(path) for path in sorted(paths)]
        if any(f is None for f in folds):
            print(f'Could not compute for ({model, split})')
            continue

        for i, pred_coco_json in enumerate(folds):
            ann_file = f'/datasets/vhr-silva/subsets/{split}/split_{i+1}.json'

            coco_gt = COCO(COCO.load_json(ann_file))
            coco_dt = coco_gt.loadRes(COCO.load_json(pred_coco_json))
            coco_eval = COCOeval_faster(coco_gt, coco_dt, 'segm', extra_calc=True)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            out = output_path / model
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f'split_{i}.json', 'w') as f:
                obj = coco_eval.stats_as_dict
                json.dump(obj, f)
