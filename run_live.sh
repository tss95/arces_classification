#!/bin/bash
# Delegate live inference to ml_array_data_classification by default
export SCRIPT_NAME=live_via_ml_array.py
export MODEL_CONFIG=alexnet.yaml
echo "Running live script"
export GPU_DEVICE="device=0"
source common.sh "$@" --predict
