#!/bin/bash
# usage python3 vq_ace/scripts/export_onnx.sh path_to_checkpoint
PATH_TO_CHECKPOINT=$1
PATH_TO_RUN=$(dirname "$(dirname "$PATH_TO_CHECKPOINT")")
SCRIPT_DIR=$(dirname "$(realpath $0)")
python3 ${SCRIPT_DIR}/export_onnx.py  --config-path="$PATH_TO_RUN/hydra" \
    --config-name=config +resume_path="$PATH_TO_CHECKPOINT" \
    dataset_cfg.data.data_directory=/home/chenyu/Data/generated/rokoko/retargeted/ \
    output_dir=${PATH_TO_RUN}

