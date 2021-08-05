#!/bin/bash

for num_layers in 1 2 3
do
    for simulation_time_limit in 30 60 120 # Minutes
    do
        config_name="DQN_${num_layers}_${simulation_time_limit}"

        sbatch --job-name=${config_name} \
            --output=scripts/slurm_out/${config_name}_%A.out \
            --error=scripts/slurm_out/${config_name}_%A.err \
            scripts/tune.slurm scripts/tune_config/${config_name}.in
    done
done


for policy_layers in 1 2 3
do
    for critic_layers in 1 2 3
    do
        for simulation_time_limit in 30 60 120 # Minutes
        do
            config_name="DDPG_${policy_layers}_${critic_layers}_${simulation_time_limit}"

            sbatch --job-name=${config_name} \
                --output=scripts/slurm_out/${config_name}_%A.out \
                --error=scripts/slurm_out/${config_name}_%A.err \
                scripts/tune.slurm scripts/tune_config/${config_name}.in
        done
    done
done
