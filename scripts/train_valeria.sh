#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu_96h
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=william.guimont-martin@norlab.ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --job-name=$CONFIG-$FOLD
#SBATCH --output=%x-%j.out
#SBATCH --account=def-phgig4
##SBATCH --array=0-9
#SBATCH --reservation=C7_9369

# ARGS: FOLD, CONFIG

# Prepare tmp directory
cd "$SLURM_TMPDIR" || exit 1
cp -r "$HOME"/ultralytics/ ./
cp -r "$HOME"/projects/def-phgig4/silva_vhr/yolo.sif ./
cp -r "$HOME"/projects/def-phgig4/silva_vhr/vhr-silva-yolo.zip ./vhr-silva.zip
cp -r "$HOME"/projects/def-phgig4/silva_vhr/hg ./hg || mkdir hg
cp -r "$HOME"/projects/def-phgig4/silva_vhr/*.pt ./ultralytics/

# Prepare dataset
unzip -q vhr-silva.zip

# Run the sweep
module load apptainer
module load httpproxy

cd ultralytics || exit 1
export WANDB_MODE=online
export PYTHONPATH=.
apptainer exec --nv --bind "$SLURM_TMPDIR"/vhr-silva:/datasets/vhr-silva --bind "$SLURM_TMPDIR"/hg:/hg --env "WANDB_MODE=online" ../yolo.sif \
  bash -c "python3 -m venv venv; source venv/bin/activate; pip install -e '.[dev]'; pip install -r requirements.txt; PYTHONPATH=. python ultralytics/train.py --data=/datasets/vhr-silva-yolo/$FOLD --hp configs/$CONFIG --outdir $HOME/projects/def-phgig4/silva_vhr/yolo_runs"

