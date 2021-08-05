#!/bin/bash

for config_name in "DDPG_2" "DQN_2"
do
    sbatch --job-name=${config_name} --array=0-$(($1-1)) \
        --output=scripts/slurm_out/${config_name}_%A_%a.out \
        --error=scripts/slurm_out/${config_name}_%A_%a.err \
        scripts/train.slurm scripts/train_config/${config_name}.in \
        scripts/train_config/${config_name}.out
done
