#!/bin/bash
#
# submission.sh: The script to be launched in the Docker image.
#
# Usage: submission.sh <input_data_pattern> <output_file>
#   input_data_pattern: jsonl.gz NQ evaluation files,
#   output_file: json file containing answer key produced by the model.
#
# Sample usage:
#   submission.sh nq-dev-0?.jsonl.gz predictions.json

set -e
set -x

INPUT_PATH=$1
OUTPUT_PATH=$2

cd /nq_model
python3 -m run_nq \
  --logtostderr \
  --bert_config_file="/nq_model/model/bert_config.json" \
  --vocab_file="/nq_model/model/vocab-nq.txt" \
  --init_checkpoint="/nq_model/model/bert_joint.ckpt" \
  --output_dir="/tmp" \
  --do_predict \
  --predict_file="$INPUT_PATH" \
  --output_prediction_file="$OUTPUT_PATH"