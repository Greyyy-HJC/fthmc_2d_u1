#!/bin/bash -l

#PBS -N fthmc_compare
#PBS -A fthmc
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=12:00:00
#PBS -o /eagle/fthmc/run/fthmc_2d_u1/evaluation_test/logs/compare_fthmc_rsat_L64_lr0.001_wd0.0001_init0.001.log

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/fthmc_2d_u1/evaluation_test
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
echo "Python path: $(which python)"
export PYTHONPATH="/eagle/fthmc/run"
echo "PYTHONPATH: $PYTHONPATH"

# run
python compare_fthmc.py --lattice_size 64 --n_configs 4096 --beta 6.0 --train_beta 4.0 --step_size 0.06 --ft_step_size 0.05 --max_lag 200 --rand_seed 2008 --model_tag 'rsat' --save_tag 'rsat_L32_lr0.001_wd0.0001_init0.001' --device 'cuda'

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
