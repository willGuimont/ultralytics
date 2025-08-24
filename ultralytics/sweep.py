import argparse
import os
import pickle

from ultralytics import YOLO


def build_search_space():
    """Build a Ray Tune search space inspired by default args, keeping disabled augs fixed at 0.0."""
    # Import locally to avoid hard dependency when not using Ray
    from ray import tune

    return {
        # Optimizer and LR schedule
        "lr0": tune.loguniform(1e-4, 5e-2),  # center near 0.01
        "lrf": tune.loguniform(1e-3, 0.2),   # final LR ratio (default 0.01)
        "momentum": tune.uniform(0.85, 0.99),
        "weight_decay": tune.loguniform(1e-6, 5e-3),
        "warmup_epochs": tune.uniform(0.0, 5.0),
        "warmup_momentum": tune.uniform(0.6, 0.95),
    "warmup_bias_lr": tune.loguniform(1e-4, 3e-1),  # default 0.1

        # Loss gains
        "box": tune.uniform(3.0, 10.0),   # default 7.5
        "cls": tune.uniform(0.2, 2.0),    # default 0.5
    "dfl": tune.uniform(0.5, 3.0),    # default 1.5

        # Color augs (fixed to args.yaml values to match working training)
        "hsv_h": tune.uniform(0.0, 0.05),
        "hsv_s": tune.uniform(0.3, 0.9),
        "hsv_v": tune.uniform(0.2, 0.8),

        # Geometric augs (fixed to args.yaml values)
        "translate": tune.uniform(0.0, 0.3),
        "scale": tune.uniform(0.2, 0.8),
        "fliplr": 0.5,
        "mosaic": 1.0,

        # Keep disabled augmentations OFF (fixed constants from defaults)
        "degrees": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "bgr": 0.0,
        "auto_augment": "randaugment",
        "copy_paste_mode": "flip",
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO sweep script')
    parser.add_argument('--size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n, s, m, l, or x (default: s)')
    args = parser.parse_args()
    size = args.size

    model = YOLO(f'yolo11{size}-seg.pt')

    # Ray Tune search space honoring disabled augs
    space = build_search_space()

    result_grid = model.tune(
        imgsz=[1456,1092],
        iterations=100,
        data='/datasets/vhr-silva/forests-8_kfold_5.yaml',
        use_ray=True,
        space=space,
        epochs=300,
        grace_period=10,
        project=f'yolo11{size}-sweep',
        wandb_project=f'yolo11{size}-sweep',
        name=f'yolo11{size}-sweep',
        gpu_per_trial=1
    )
    print(result_grid)

    output_dir = os.path.join(os.path.expanduser('~'), f'yolo11{size}-sweep')
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'result_grid.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result_grid, f)
    print(f'Results saved to {result_path}')
