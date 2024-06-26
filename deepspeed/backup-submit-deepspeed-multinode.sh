#!/bin/bash
#SBATCH --job-name=multinode-video-gen
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --output=video_gen_deepspeed_output.log  # Add this line to specify the output file

export GPUS_PER_NODE=1
export OMP_NUM_THREADS=1

# Activate the conda environment
source ~/miniconda3/bin/activate
conda activate videogen

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(( RANDOM % (50000 - 30000 + 1 ) + 30000 ))
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)


# AWS specific
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_CPP_LOG_LEVEL=INFO
export CUDA_LAUNCH_BLOCKING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export LOGLEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1


# Create a custom hostfile
# Get the list of allocated nodes
nodes=$(scontrol show hostname $SLURM_JOB_NODELIST)

hostfile="hostfile-distribution-${SLURM_JOB_ID}"
for node in $nodes; do
  echo "$node slots=$GPUS_PER_NODE" >> $hostfile
done

srun accelerate launch \
             --config_file accelerate_config.yaml \
	         --main_process_ip ${MASTER_ADDR} \
             --main_process_port ${MASTER_PORT} \
             --machine_rank $SLURM_NODEID \
             --num_processes $NUM_PROCESSES \
             --num_machines $NNODES \
             --machine_rank $SLURM_PROCID \
             --use_deepspeed \
             --deepspeed_multinode_launcher standard \
             --deepspeed_config_file ds_config.json \
             --deepspeed_hostfile $hostfile \
             --rdzv_conf "rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT rdzv_backend=c10d" \
            train_stage_1.py --config configs/train/stage1.yaml

