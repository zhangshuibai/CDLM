#!/bin/bash

set -x

export TOKENIZERS_PARALLELISM=false

NNODES=${NNODES:=1}
NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}


torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT $@ 2>&1 | tee log.txt
