#!/usr/bin/env bash

set -x

export PYTHONPATH=/ceph-fj/open-source/speechbrain:$PYTHONPATH
export PYTHONPATH=/ceph-fj/open-source/k2/k2/python:$PYTHONPATH
export PYTHONPATH=/ceph-fj/open-source/k2/build_release-1.7.1/lib:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=1

python3 ./main.py ./params.yaml
