#!/bin/bash
#
# Script launcher for A100 nodes.
#
# Example:
#     sbatch --nodes=2 --gpus-per-node=8 --cpus-per-task=114 --partition=compute ./scripts/training/rlhf/run_rlhf_combine.sh
#
# Nodes selection can be done via running `sinfo -lN`
# and then specifying the node names directly by adding
# `--nodelist=<node1>,<node2>` to the sbatch command above.
#
# Check your task via `squeue -u $USER` and cancel it via `scancel <job_id>` if needed.
#
# Set script flags down below manually as you need.
#
#SBATCH --job-name=rlhf_training
#SBATCH --nodes=2  # adjust as needed
#SBATCH --ntasks-per-node=1
#SBATCH --output=./rlhf_experiments/slurm_%j/log.log
#SBATCH --exclusive


####### SLURM setup #######

export PMI_DEBUG=1

cd ./rlhf_experiments/slurm_$SLURM_JOB_ID

MACHINEFILE="hostfile"
ORDEREDMACHINEFILE="ordered_hostfile_system_name"
ORDEREDRANKMACHINEFILE="rankfile_system_name"
scontrol show hostnames $SLURM_JOB_NODELIST >$MACHINEFILE

echo "SLURM_JOB_NUM_NODES = " $SLURM_JOB_NUM_NODES
echo "SLURM_GPUS_PER_NODE = " $SLURM_GPUS_PER_NODE
echo "SLURM_NTASKS_PER_NODE = " $SLURM_NTASKS_PER_NODE

echo MACHINEFILE
cat $MACHINEFILE

source /etc/os-release
python3 /home/$USER/node_ordering_by_rack.py --input_file $MACHINEFILE >/dev/null

echo ORDEREDMACHINEFILE
cat $ORDEREDMACHINEFILE
echo ORDEREDRANKMACHINEFILE
cat $ORDEREDRANKMACHINEFILE

mpivars_path=$(ls /usr/mpi/gcc/openmpi-*/bin/mpivars.sh)

if [[ "$mpivars_path" == "" ]]; then
    mpivars_path=$(ls /opt/openmpi-*/bin/mpivars.sh)
fi

if [[ "$mpivars_path" == "" ]]; then
    echo "Could not find MPIPATH"
    exit
fi

source $mpivars_path

shape=$(curl -sH "Authorization: Bearer Oracle" -L http://169.254.169.254/opc/v2/instance/ | jq .shape)
if [ $shape == \"BM.GPU.B4.8\" ] || [ $shape == \"BM.GPU.A100-v2.8\" ]; then
    var_UCX_NET_DEVICES=mlx5_0:1
    var_NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
elif [ $shape == \"BM.GPU4.8\" ]; then
    var_UCX_NET_DEVICES=mlx5_4:1
    var_NCCL_IB_HCA="=mlx5_0,mlx5_2,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14,mlx5_16,mlx5_1,mlx5_3,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15,mlx5_17"
fi

export NCCL_DEBUG=WARN \
    NCCL_IB_QPS_PER_CONNECTION=4 \
    NCCL_IB_GID_INDEX=3 \
    NCCL_IB_TC=41 \
    NCCL_IB_SL=0 \
    HCOLL_ENABLE_MCAST_ALL=0 \
    coll_hcoll_enable=0 \
    UCX_TLS=ud,self,sm \
    UCX_NET_DEVICES=${var_UCX_NET_DEVICES} \
    NCCL_ALGO=Ring \
    NCCL_IB_HCA="${var_NCCL_IB_HCA}"

####### End of SLURM setup #######

ROOT_DIR=/home/$USER/tts_training
cd $ROOT_DIR
source .venv/bin/activate

# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 1 node for training and the 2nd node for vLLM
TRAIN_NODE="${NODELIST[0]}"  # Node 0 for training
VLLM_NODE="${NODELIST[1]}"  # Node 1 for vLLM

export WANDB_PROJECT="RLHF_Finch"
export PYTHONPATH=$PYTHONPATH:$(pwd)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Increase memory fraction allowed for CUDA operations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

# The base model for vLLM, use the same checkpoint in the training config: "rlhf_training".base_model_dir
VLLM_MODEL_PATH="/home/feifan/tts_training/rlhf_experiments/rlhf_combine_1102/checkpoint-11500"

srun --nodes=1 --ntasks-per-node=1 --gpus-per-task=$SLURM_GPUS_PER_NODE --nodelist=$TRAIN_NODE \
    accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --num_processes=$SLURM_GPUS_PER_NODE \
    --machine_rank=${SLURM_PROCID} \
    --main_process_ip=$TRAIN_NODE \
    --rdzv_backend=c10d \
    $ROOT_DIR/scripts/training/rlhf/rlhf_main.py \
    --use_wandb \
    --config_path=$ROOT_DIR/configs/finch_1b_rlhf_combine.json \
    --run_name=rlhf_combine_$SLURM_JOB_ID \
    --vllm_server_host=$VLLM_NODE &

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $VLLM_MODEL_PATH --tensor_parallel_size 8 --max_model_len 3072 &
wait
