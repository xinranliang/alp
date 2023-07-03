#/bin/bash

# number of gpus to use: --nproc_per_node

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
CUDA_VISIBLE_DEVICES=$1 \
python -u -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $2 \
    src/pretrain/run_random.py \
    --exp-config configs/pretrain/random.yaml \
    --run-type train
