#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00
#SBATCH --job-name=yolo-16
#SBATCH --output=%x-%j.out

cd $HOME/ultralytics || exit 1

export WANDB_MODE=online
export PYTHONPATH=.
apptainer exec --nv --bind /data/vhr-silva:/datasets/vhr-silva --env "WANDB_MODE=online" --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ../yolo.sif \
  bash -c "python3 -m venv venv; source venv/bin/activate; pip install -e '.[dev]'; pip install -r requirements.txt; PYTHONPATH=. python ultralytics/sweep.py --size $SIZE --iterations $ITER --split $SPLIT --model $MODEL"
#podman run --gpus all --rm --ipc host \
#      -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#      -e WANDB_API_KEY=$WANDB_API_KEY \
#      -e SLURM_JOB_ID=$SLURM_JOB_ID \
#  	--mount type=bind,source=$(pwd)/,target=/code \
#      --mount type=bind,source=/data/vhr-silva,target=/datasets/vhr-silva \
#  	--mount type=bind,source=/dev/shm,target=/dev/shm \
#  	--mount type=bind,source=$HOME/.config/Ultralytics,target=/root/.config/Ultralytics \
#  	--env "WANDB_MODE=online" \
#  	pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime  bash -c "cd /code; python3 -m venv venv; source venv/bin/activate; pip install -e '.[dev]'; pip install -r requirements.txt; PYTHONPATH=. python ultralytics/sweep.py --size $SIZE --iterations $ITER --split $SPLIT"

# container_id=$(
#   podman run -d --gpus all --rm --ipc host \
#       -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
#       -e WANDB_API_KEY=$WANDB_API_KEY \
#       -e SLURM_JOB_ID=$SLURM_JOB_ID \
#   	--mount type=bind,source=$HOME/projects/vhr-mask2former,target=/code \
#       --mount type=bind,source=/data/vhr-silva,target=/datasets/vhr-silva \
#   	--mount type=bind,source=/dev/shm,target=/dev/shm \
#   	vhr-mask2former python image_m2f/sweep/sweep.py \
#   		--sweep-id "$SWEEP_ID" \
#           --dataset-name silva-vhr \
#           --subset-name forests
# )
# 
# stop_container() {
#   podman logs $container_id
#   echo "Stopping container $container_id..."
#   podman container stop $container_id
# }
# 
# trap stop_container EXIT
# echo "Container started with ID: $container_id"
# podman wait $container_id || stop_container
