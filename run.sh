#!/bin/bash
#export SCRIPT_NAME=multi_gpu_test.py
export SCRIPT_NAME=code_test.py

export MODEL_CONFIG=alexnet.yaml
echo "Running train script"
export GPU_DEVICE="device=0"
source common.sh "$@"