#!/bin/bash

python inference/baseline.py
python inference/chain_of_thought.py
python inference/self_consistency.py
python inference/adaptive.py

