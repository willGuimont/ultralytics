# Valeria setup

## Upload the datasets
Send the data to Valeria using [willGuimont/globus-docker](https://github.com/willGuimont/docker-globus).

## Build the container

To train our models on Valeria, we use [Apptainer](https://apptainer.org/).
Use the following commands to build the container:

```shell
apptainer build --sandbox yolo yolo.def
# Test to make sure the sandbox is working
apptainer shell --nv --writable yolo
# Need to install the requirements
pip install -e '.[dev]'
pip install -r requirements.txt

# Build the SIF file
apptainer build yolo.sif yolo/
# Test the SIF file
apptainer shell --nv yolo.sif
```

## Test it on Valeria

```shell
# No internet access
salloc --time=120:00 --cpus-per-task=32 --mem=64G --partition=gpu --gres=gpu:a100:1 --account=def-phgig4
# With internet access
salloc --time=120:00 --cpus-per-task=32 --mem=64G --partition=gpu_inter --gres=shard:1 --account=def-phgig4
```

## Running the sweep

Using [willGuimont/sjm](https://github.com/willGuimont/sjm).

```shell
# Generate the sweep from the config file
python feature_m2f/sweep/generate.py
# Add SWEEP_ID to your .env file
source .env
cd scripts/
sjm add valeria username@login.valeria.science
sjm pull valeria vhr_mask2former
sjm run valeria sweep_valeria.sh SWEEP_ID=$SWEEP_ID
```
