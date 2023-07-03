#/bin/bash

# number of gpus per node / master process: --nproc_per_node
# number of nodes / master process want to be launched: --nnodes
# --use_env \

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
CUDA_VISIBLE_DEVICES=$1 \
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $2 \
    src/pretrain/run_ddppo.py \
    --exp-config configs/pretrain/rnd_alp.yaml \
    --run-type train