#!/bin/bash -l

#PBS -N train
#PBS -A fthmc
#PBS -l select=2
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -o /eagle/fthmc/run/fthmc_2d_u1/ft_train_test/logs/train_L64_b2.0-b2.0_test_stablev6.log

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/fthmc_2d_u1/ft_train_test
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

# run train.py
torchrun --standalone --nproc_per_node=2 train.py --lattice_size 64 --min_beta 2.0 --max_beta 2.0 --beta_gap 0.5 --n_epochs 32 --batch_size 32 --n_subsets 8 --n_workers 0 --model_tag 'stablev6' --save_tag 'stablev6' --rand_seed 2008 --if_identity_init

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
