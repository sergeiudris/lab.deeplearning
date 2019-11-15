#!/bin/bash

set -evx

export_bert(){
  # regression classification question_answering
  python3 /opt/root/python/bert/export/export.py \
    --task classification \
    --prefix "bert-cls-4" \
    --seq_length 512 \
    --num_classes 4 \
    --output_dir /opt/app/tmp/data/bert-base
}


"$@"