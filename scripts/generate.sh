#!/bin/bash

time_limit=20 # Hours
train_num_episodes=5
eval_num_episodes=20
max_episode_length=10000

for num_layers in 1 2 3
do
    for simulation_time_limit in 30 60 120 # Minutes
    do
        script_name="scripts/tune_config/DQN_${num_layers}_${simulation_time_limit}"

        echo '{"hparams":
            {
                "learning_rate":
                    {
                        "min": -15,
                        "max": -5
                    },
                "layer":
                    {
                        "min": 0,
                        "max": 8,
                        "num": '$num_layers'
                    },
                "epsilon":
                    {
                        "min": 0,
                        "max": 0.15
                    },
                "batch_size":
                    {
                        "min": 3,
                        "max": 9
                    }
            },
            "agent_name": "DQN",
            "strategist_name": "BasicStrategist",
            "simulation_time_limit":  '$((simulation_time_limit*60))',
            "time_limit": '$((time_limit*60*60))',
            "max_episode_length": '$max_episode_length',
            "train_num_episodes": '$train_num_episodes',
            "eval_num_episodes": '$((simulation_time_limit/2))',
            "checkpoint": "'$script_name'.pkl"
        }' >> "${script_name}.in"
    done
done


for policy_layers in 1 2 3
do
    for critic_layers in 1 2 3
    do
        for simulation_time_limit in 30 60 120 # Minutes
        do
            script_name="scripts/tune_config/DDPG_${policy_layers}_${critic_layers}_${simulation_time_limit}"

            echo '{"hparams":
            {
                "policy_learning_rate":
                    {
                        "min": -20,
                        "max": -3
                    },
                "critic_learning_rate":
                    {
                        "min": -20,
                        "max": -3
                    },
                "policy_layer":
                    {
                        "min": 0,
                        "max": 8,
                        "num": '$policy_layers'
                    },
                "critic_layer":
                    {
                        "min": 0,
                        "max": 8,
                        "num": '$critic_layers'
                    },
                "sigma":
                    {
                        "min": 0,
                        "max": 0.5
                    },
                "batch_size":
                    {
                        "min": 3,
                        "max": 9
                    }
            },
            "agent_name": "DDPG",
            "strategist_name": "BasicStrategist",
            "simulation_time_limit":  '$((simulation_time_limit*60))',
            "time_limit": '$((time_limit*60*60))',
            "max_episode_length": '$max_episode_length',
            "train_num_episodes": '$train_num_episodes',
            "eval_num_episodes": '$((simulation_time_limit/2))',
            "checkpoint": "'$script_name'.pkl"
        }' >> "${script_name}.in"
        done
    done
done
