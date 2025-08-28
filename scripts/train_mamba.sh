#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00
#SBATCH --job-name=$CONFIG-$FOLD
#SBATCH --output=%x-%j.out

# ARGS: FOLD, CONFIG

cd $HOME/ultralytics || exit 1

export WANDB_MODE=online
export PYTHONPATH=.
apptainer exec --nv --bind /data/vhr-silva:/datasets/vhr-silva --env "WANDB_MODE=online" --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ../yolo.sif \
  bash -c "python3 -m venv venv; source venv/bin/activate; pip install -e '.[dev]'; pip install -r requirements.txt; PYTHONPATH=. python ultralytics/train.py --data=/datasets/vhr-silva-yolo/$FOLD --hp configs/$CONFIG --outdir /data/yolo_runs"
