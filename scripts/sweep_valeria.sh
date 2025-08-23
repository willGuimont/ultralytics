#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu_96h
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-user=william.guimont-martin@norlab.ulaval.ca
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --job-name=yolo-small-sweep
#SBATCH --output=%x-%j.out
#SBATCH --account=def-phgig4
##SBATCH --array=0-9
#SBATCH --reservation=C7_9369

# Prepare tmp directory
cd "$SLURM_TMPDIR" || exit 1
cp -r "$HOME"/vhr-mask2former/ ./
cp -r "$HOME"/projects/def-phgig4/silva_vhr/vhr_mask2former.sif ./
cp -r "$HOME"/projects/def-phgig4/silva_vhr/vhr-silva.zip ./
cp -r "$HOME"/projects/def-phgig4/silva_vhr/hg ./hg || mkdir hg

# Prepare dataset
unzip -q vhr-silva.zip

# Run the sweep
module load apptainer
module load httpproxy

cd vhr-mask2former || exit 1
export WANDB_MODE=online
apptainer exec --nv --bind "$SLURM_TMPDIR"/vhr-silva:/datasets/vhr-silva --bind "$SLURM_TMPDIR"/hg:/hg --env "WANDB_MODE=online" ../vhr_mask2former.sif \
  bash -c "HF_HOME=/hg PYTHONPATH=. python vhr_mask2former/image_m2f/sweep/sweep.py \
    --sweep-id "$SWEEP_ID" \
    --dataset-name silva-vhr \
    --subset-name forests"
