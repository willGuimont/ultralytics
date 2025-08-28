import re
from collections import defaultdict
from pathlib import Path
import subprocess

if __name__ == '__main__':
    root = Path('/datasets/vhr-silva-yolo')
    splits = sorted(f for f in root.iterdir() if f.is_file() and 'split' not in f.name and 'binary' not in f.name)
    splits_per_size = defaultdict(list)

    regex = r'[a-z0-9\(\)\*_]*-([a-z0-9\(\)\*_]*)_kfold_[1-5].yaml'
    regex = re.compile(regex)
    for s in splits:
        split_num = re.search(regex, s.name)[1]
        splits_per_size[split_num].append(s)

    for k, v in splits_per_size.items():
        assert len(v) == 5, f'{k} is missing some folds (found {len(v)})'

    configs_path = Path('./configs')
    configs = sorted(f for f in configs_path.glob('*.json'))

    regex = r'[a-z0-2]*-([0-9]+)\.json'
    regex = re.compile(regex)
    for config in configs:
        split = re.search(regex, config.name)[1]
        folds = splits_per_size[split]
        # FOLD, CONFIG
        for fold in folds:
            print(f'sjm run valeria scripts/train_valeria.sh FOLD={fold.name} CONFIG={config.name}')
