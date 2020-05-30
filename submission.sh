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
python3 -m run_nq_ensemble_modified \
  --max_seq_length=512 \
  --doc_stride=256 \
  --max_contexts=48 \
  --output_dir="/nq_model/output" \
  --predict_file="$INPUT_PATH"  \
  --final_output_prediction_file="$OUTPUT_PATH"