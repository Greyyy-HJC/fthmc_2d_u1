#!/bin/bash -l

#PBS -N gen
#PBS -A fthmc
#PBS -l select=1
#PBS -l filesystems=home:eagle
#PBS -q by-gpu
#PBS -j oe
#PBS -l walltime=24:00:00
#PBS -o /eagle/fthmc/run/fthmc_2d_u1/gauge_gen/logs/gen_L64_b2.5-b5.5.log

# switch to the submit directory
WORKDIR=/eagle/fthmc/run/fthmc_2d_u1/gauge_gen
cd $WORKDIR

# output node info
echo ''
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
echo "Python path: $(which python)"

# check python path
export PYTHONPATH="/eagle/fthmc/run"
echo "PYTHONPATH: $PYTHONPATH"


# run conf_gen.py
python conf_gen.py --lattice_size 32 --beta 2.5 --n_thermalization 600 --store_interval 20 --n_configs 4096

python conf_gen.py --lattice_size 32 --beta 3.5 --n_thermalization 600 --store_interval 20 --n_configs 4096

python conf_gen.py --lattice_size 32 --beta 4.5 --n_thermalization 600 --store_interval 30 --n_configs 4096

python conf_gen.py --lattice_size 32 --beta 5.5 --n_thermalization 600 --store_interval 30 --n_configs 4096

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
