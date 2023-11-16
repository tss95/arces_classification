#!/bin/bash
export SCRIPT_NAME=train.py
export MODEL_CONFIG=alexnet.yaml
echo "Running train script"
export GPU_DEVICE="device=0"
source common.sh "$@"