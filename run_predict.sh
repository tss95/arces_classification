#!/bin/bash
export SCRIPT_NAME=predict.py
export MODEL_CONFIG=alexnet.yaml
echo "Running prediction script"
export GPU_DEVICE="device=0"
source common.sh "$@"