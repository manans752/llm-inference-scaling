#!/bin/bash

./venv/bin/python run_experiment.py \
  --dataset-path data/tasks.json \
  --dataset-name prototype_v1 \
  --strategies baseline chain_of_thought adaptive self_consistency
