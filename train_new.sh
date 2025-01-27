#!/bin/sh
PARTITION=Segmentation

dataset=$1
exp_name=$2
exp_dir=exp/${dataset}/${exp_name}_alm
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_alm.yaml

mkdir -p ${model_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_new.sh train_new.py ${config} ${exp_dir}

python3 -u train_new.py --config=${config} 2>&1 | tee ${result_dir}/train-$now.log
