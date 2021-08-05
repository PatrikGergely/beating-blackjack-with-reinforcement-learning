#!/bin/bash
# Input: {#JobName} {#OfNodes} {#Partition} {#ConfigPath}

if [[ $3 == *"cpu"* ]]
then
    sbatch --job-name=${1} --ntasks-per-node=20 --array=0-$(($2-1)) \
        --output=scripts/slurm_out/${1}_%A_%a.out \
        --error=scripts/slurm_out/${1}_%A_%a.err \
        --partition=cpu-batch scripts/simulate.slurm ${4}
else
    sbatch --job-name=${1} --ntasks-per-node=6 --array=0-$(($2-1)) \
        --output=scripts/slurm_out/${1}_%A_%a.out \
        --error=scripts/slurm_out/${1}_%A_%a.err \
        --partition=desktop-batch scripts/simulate.slurm ${4}
fi
