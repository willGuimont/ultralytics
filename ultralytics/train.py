import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def load_hp(path: Path):
    with open(path, 'r') as f:
        hp = json.load(f)
    ignore_keys = {'node_ip', 'project', 'hostname', 'trial_id', 'trial_log_path', 'pid', 'date'}
    output = {}
    for k, v in hp.items():
        if k.startswith('_') or k in ignore_keys:
            continue
        output[k] = v.get('value', v)
    return output


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model.')
    parser.add_argument('--data', required=True, help='Path to dataset YAML.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs. If not set, will use value from hyperparameters if present.')
    parser.add_argument('--hp', required=True, help='Path to json file containing all hyperparameters.')
    parser.add_argument('--outdir', required=True, help='Output directory')
    args = parser.parse_args()

    hyp = load_hp(args.hp)

    hyp['project'] = Path(args.outdir)
    config_name = Path(args.hp).name
    data_name = Path(args.data).name
    hyp['name'] = f"{config_name}_{data_name}"

    if args.data:
        hyp['data'] = args.data
    if args.epochs:
        hyp['epochs'] = args.epochs
    if 'model' not in hyp:
        assert args.model is not None, "Not model specified"
        hyp['model'] = args.model
    if args.epochs:
        hyp['epochs'] = args.epochs

    print('Training with hyperparameters:', hyp)
    model = YOLO(hyp['model'])
    results = model.train(**hyp)
    print('Training results:', results)


if __name__ == '__main__':
    main()
