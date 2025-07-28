#!/bin/bash -l

#PBS -N tune
#PBS -A fthmc
#PBS -l select=2
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -o /eagle/fthmc/run/fthmc_2d_u1/ft_train_tune/logs/train_L32_b2.0-b2.0_lite_tuned_with_init.log

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/fthmc_2d_u1/ft_train_tune
cd $WORKDIR

# output node info
echo ' '
echo ">>> PBS_NODEFILE content:"
cat $PBS_NODEFILE
NODES=$(cat $PBS_NODEFILE | uniq | wc -l)
TASKS=$(wc -l < $PBS_NODEFILE)
echo "${NODES}n*${TASKS}t"

# Get GPU info
nvidia-smi
nvcc --version

# show current time
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: $start_time"

# Initialize conda properly
source /eagle/fthmc/env/py_env.sh

# check python version
python --version

# check python path
export PYTHONPATH="/eagle/fthmc/run"
echo "Python path: $(which python)"
echo "PYTHONPATH: $PYTHONPATH"

# set pytorch cuda alloc config
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True


# lr in [0.01, 0.005, 0.001, 0.0005, 0.0001], pick 0.005
# weight_decay in [0.01, 0.001, 0.0001, 0.00001], pick 0.001
# init_std in [0.1, 0.01, 0.001, 0.0001] pick 0.001

# run train.py
torchrun --standalone --nproc_per_node=2 train.py \
    --lattice_size 32 --min_beta 2.0 --max_beta 2.0 --beta_gap 1.0 \
    --n_epochs 64 --batch_size 32 --n_subsets 8 --n_workers 0 \
    --model_tag 'lite' --save_tag 'lite_L32_tuned_with_init' --rand_seed 2008 --if_identity_init \
    # --continue_beta 2.0 
    # --lr 0.005 --weight_decay 0.001 --init_std 0.0001 \
    

# calculate total time
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "End time: $end_time"

# total time
start_seconds=$(date --date="$start_time" +%s)
end_seconds=$(date --date="$end_time" +%s)
duration=$((end_seconds - start_seconds))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))
echo "Total time: ${hours}h ${minutes}m ${seconds}s"
