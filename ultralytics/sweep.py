import argparse
import os
from datetime import datetime

from ultralytics import YOLO


def build_search_space():
    from ray import tune

    return {
        # Optimizer and LR schedule
        "lr0": tune.uniform(1e-5, 1e-1),
        "lrf": tune.uniform(0.01, 1.0),
        "momentum": tune.uniform(0.6, 0.99),
        "weight_decay": tune.uniform(0.0, 0.001),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.0, 0.95),

        # Loss gains
        "box": tune.uniform(3.0, 10.0),  # default 7.5
        "cls": tune.uniform(0.2, 2.0),  # default 0.5
        "dfl": tune.uniform(0.5, 3.0),  # default 1.5

        # Color augs (fixed to args.yaml values to match working training)
        "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
        "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
        "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)

        # Do not modify, otherwise some augmentations might crash
        "mosaic": 1.0,

        # Geometric augs (fixed to args.yaml values)
        "degrees": tune.uniform(0.0, 45.0),  # image rotation (+/- deg)
        "translate": tune.uniform(0.0, 0.9),  # image translation (+/- fraction)
        "scale": tune.uniform(0.0, 0.9),  # image scale (+/- gain)
        "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
        "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
        "flipud": tune.uniform(0.0, 1.0),  # image flip up-down (probability)
        "fliplr": tune.uniform(0.0, 1.0),  # image flip left-right (probability)
        "bgr": tune.uniform(0.0, 1.0),  # image channel BGR (probability)
        "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
        "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
        "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO sweep script')
    parser.add_argument('--size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n, s, m, l, or x (default: s)')
    parser.add_argument('--model', type=int, default=11, help='Model version')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of hyperparameter tuning iterations (default: 100)')
    parser.add_argument('--split', type=int, default=8, help='Image downscaling')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')

    args = parser.parse_args()
    size = args.size
    model_version = args.model
    iterations = args.iterations
    split = args.split
    epochs = args.epochs

    if model_version == 11:
        pt = f'yolo11{size}-seg.pt'
    elif model_version == 12:
        pt = f'yolov12{size}-seg.pt'
    else:
        raise ValueError(f'Invalid model {model_version}')

    model = YOLO(pt)
    space = build_search_space()

    # Base tuning kwargs
    name = f'yolo{model_version}{size}-sweep-split-{split}'
    tune_kwargs = dict(
        imgsz=[1092, 1456],
        iterations=iterations,
        data=f'/datasets/vhr-silva/forests-{split}_kfold_5.yaml',
        use_ray=True,
        space=space,
        epochs=epochs,
        project=name,
        wandb_project=name,
        name=name,
        gpu_per_trial=1,
        save=False
    )

    if size == 'x':
        tune_kwargs['batch'] = 1

    if split == 16:
        tune_kwargs['imgsz'] = [546, 728]
    elif split == 32:
        tune_kwargs['imgsz'] = [273, 364]

    result_grid = model.tune(**tune_kwargs)
    print(result_grid)

    # Save the string representation of result_grid to a timestamped directory in $HOME (no pickling)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(os.path.expanduser('~'), f'yolo11{size}-sweep-{split}-{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'result_grid.txt')
    with open(result_path, 'w') as f:
        f.write(str(result_grid))
    print(f'Results saved to {result_path}')
