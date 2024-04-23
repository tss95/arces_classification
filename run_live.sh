#!/bin/bash
export SCRIPT_NAME=gbf_iter.py
export MODEL_CONFIG=alexnet.yaml
echo "Running live script"
export GPU_DEVICE="device=0"
source common.sh "$@" --predict