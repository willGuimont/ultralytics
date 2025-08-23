import argparse
import os
import pickle

from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO sweep script')
    parser.add_argument('--size', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n, s, m, l, or x (default: s)')
    args = parser.parse_args()
    size = args.size

    model = YOLO(f'yolo11{size}-seg.pt')
    result_grid = model.tune(
        imgsz=[1456,1092],
        iterations=500,
        data='/datasets/vhr-silva/kfold_1.yaml',
        use_ray=True,
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
