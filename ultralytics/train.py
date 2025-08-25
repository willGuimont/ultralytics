import argparse
import ast
from ultralytics import YOLO

def load_best_hyperparameters(result_path):
    """
    Loads the best hyperparameters from a result_grid.txt file.
    Assumes the file contains a string representation of a dict or object with a 'best_config' or similar key.
    """
    with open(result_path, 'r') as f:
        content = f.read()
    # Try to safely evaluate the string to a dict
    try:
        result = ast.literal_eval(content)
    except Exception:
        raise RuntimeError("Could not parse result_grid.txt. Please check its format.")
    # Try common keys for best config
    for key in ['best_config', 'best', 'best_params', 'best_hyperparameters']:
        if key in result:
            return result[key]
    # If the result itself is a dict of hyperparameters
    if isinstance(result, dict):
        return result
    raise RuntimeError("Could not find best hyperparameters in result_grid.txt.")

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model with best hyperparameters from sweep.')
    parser.add_argument('--result', type=str, default='result_grid.txt',
                        help='Path to result_grid.txt containing best hyperparameters.')
    parser.add_argument('--model', type=str, default='yolo11s-seg.pt',
                        help='YOLO model checkpoint to use.')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset YAML. If not set, will use value from hyperparameters if present.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs. If not set, will use value from hyperparameters if present.')
    args = parser.parse_args()

    best_hyp = load_best_hyperparameters(args.result)

    # Allow override of data/epochs from CLI
    if args.data:
        best_hyp['data'] = args.data
    if args.epochs:
        best_hyp['epochs'] = args.epochs
    if 'model' not in best_hyp:
        best_hyp['model'] = args.model

    print('Training with hyperparameters:', best_hyp)
    model = YOLO(best_hyp['model'])
    results = model.train(**best_hyp)
    print('Training results:', results)

if __name__ == '__main__':
    main()
