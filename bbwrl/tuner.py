# MIT License
#
# Copyright (c) 2021 Patrik Gergely
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
""" Tunes a ML based bettor model.

Use: python bbwrl/tuner.py <tune_config_file>

Example config file
{"hparams":
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
                "num": 3
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
    "simulation_time_limit":  3600,
    "time_limit": 72000,
    "max_episode_length": 10000,
    "train_num_episodes": 5,
    "eval_num_episodes": 30,
    "checkpoint": "scripts/tune_config/DQN_3_60.pkl"
}

"""


import functools
import json
import os
from pathlib import Path
import sys
from typing import Callable, Any, Dict

import ray
from ray import tune
from ray.tune.suggest import bayesopt

from bbwrl import trainer


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _get_hparams(
    config: Dict[str, Any],
    hparam_map: Dict[str, Callable[[float], float]]
) -> Dict[str, Any]:
    """ Creates hyper-parameters from a configuration.

    Maps each configuration to a hyper-parameter.
    Layer parameters are appended to a single list of shape and used as a
    hyperparameter.

    Args:
        config: The configuration generated by the optimizer.
        hparam_map: The function mapping a configuration to hyper-parameters.

    Returns:
        A dictionary containing the hyper-parameters.
    """
    hparams = {}
    network_shape = []
    policy_network_shape = []
    critic_network_shape = []

    for parameter in sorted(config):
        if 'layer' in parameter:
            value = hparam_map['layer'](config[parameter])
            if 'policy' in parameter:
                policy_network_shape.append(value)
            elif 'critic' in parameter:
                critic_network_shape.append(value)
            else:
                network_shape.append(value)
        elif 'learning_rate' in parameter:
            hparams[parameter] = hparam_map['learning_rate'](config[parameter])
        else:
            hparams[parameter] = hparam_map[parameter](config[parameter])

    if network_shape:
        hparams['network_shape'] = network_shape
    if policy_network_shape:
        hparams['policy_network_shape'] = policy_network_shape
    if critic_network_shape:
        hparams['critic_network_shape'] = critic_network_shape

    return hparams


def train_config(config: Dict[str, Any],
                 hparam_map: Dict[str, Callable[[float], float]],
                 agent_name: str,
                 strategist_name: str,
                 simulation_time_limit: int,
                 max_episode_length: int,
                 train_num_episodes: int,
                 eval_num_episodes: int) -> None:
    """ Trains, evaluates then reports the score for a configuration.

    Args:
        config: The configuration generated by the optimizer.
        hparam_map: The function mapping a configuration to hyper-parameters.
        agent_name: The name of the type of agent to train (DQN or DDPG).
        strategist_name: The name of the strategist to use.
        simulation_time_limit: The number of seconds to train the bettor for.
        max_episode_length: The number of games to terminate the episode after.
        train_num_episodes: The number of episodes to simulate the bot for at a
            time when training the bettor.
        eval_num_episodes: The number of episodes to simulate the bot for when
            evaluating.

    Returns:
        None
    """
    _, score = trainer.train_and_evaluate(agent_name,
                                          strategist_name,
                                          _get_hparams(config, hparam_map),
                                          simulation_time_limit,
                                          max_episode_length,
                                          train_num_episodes,
                                          eval_num_episodes)
    # Send the current training result back to Tune
    tune.report(score=score)


def tune_model(search_space: Dict[str, Any],
               hparam_map: Dict[str, Callable[[float], float]],
               agent_name: str,
               strategist_name: str,
               simulation_time_limit: int,
               max_episode_length: int,
               train_num_episodes: int,
               eval_num_episodes: int,
               time_limit: int,
               checkpoint: str) -> Dict[str, Any]:
    """ Tunes a model using Bayesian Optimization.

    Args:
        search_space: The search space to pick configurations from.
        hparam_map: The function mapping a configuration to hyper-parameters.
        agent_name: The name of the type of agent to train (DQN or DDPG).
        strategist_name: The name of the strategist to use.
        simulation_time_limit: The number of seconds to train the bettor for.
        max_episode_length: The number of games to terminate the episode after.
        train_num_episodes: The number of episodes to simulate the bot for at a
            time when training the bettor.
        eval_num_episodes: The number of episodes to simulate the bot for when
            evaluating.
        time_limit: The number of seconds to run the tuner for.
        checkpoint: The location to save the checkpoint for the optimizer.

    Returns:
        A dictionary containing the hyper-parameters that performed best during
        the training.

    """
    applied_train_function = functools.partial(
        train_config,
        hparam_map=hparam_map,
        agent_name=agent_name,
        strategist_name=strategist_name,
        simulation_time_limit=simulation_time_limit,
        max_episode_length=max_episode_length,
        train_num_episodes=train_num_episodes,
        eval_num_episodes=eval_num_episodes)
    num_samples = (time_limit/simulation_time_limit)
    bayesopt_search = bayesopt.BayesOptSearch(search_space,
                                              metric='score',
                                              mode='max')
    checkpoint_file = Path(checkpoint)
    if checkpoint_file.is_file():
        print('Restoring search from {}'.format(checkpoint))
        bayesopt_search.restore(checkpoint)

    analysis = tune.run(applied_train_function,
                        search_alg=bayesopt_search,
                        config = search_space,
                        resources_per_trial={'gpu': 1},
                        num_samples=round(num_samples))
    bayesopt_search.save(checkpoint)
    return analysis.get_best_config(metric='score', mode='max')


def _search_space(hparams: Dict[str, Any]) -> Dict[str, Any]:
    """ Create a search_space from hyper-parameter bounds.

    For hyper-parameters a min and max value determines what bounds to search
    between, when these are equal the hyper-parameter won't be optimized.
    When `num` is passed it generates `num` many identical hyper-parameters.

    Returns:
        A search space for configurations.
    """
    search_space = {}
    for hparam in hparams:
        hparam_values = hparams[hparam]
        hparam_min = hparam_values['min']
        hparam_max = hparam_values['max']
        interval = (hparam_min
                    if hparam_min == hparam_max
                    else (hparam_min, hparam_max))
        if 'num' in hparam_values:
            for i in range(hparam_values['num']):
                search_space[hparam+'_{}'.format(i+1)]=interval
        else:
            search_space[hparam]=interval
    return search_space


def main():
    ray.init(dashboard_port = 9001, num_cpus=1, num_gpus=1)
    with open(sys.argv[1]) as json_file:
        parameters = json.load(json_file)
    parameters['search_space'] = _search_space(parameters['hparams'])
    del parameters['hparams']
    parameters['hparam_map'] = {
        'layer': lambda x: 2**round(x),
        'learning_rate': lambda x: 10**x,
        'epsilon': lambda x: x,
        'sigma': lambda x: x,
        'batch_size': lambda x: 2**round(x),
    }
    best_config = tune_model(**parameters)
    print(_get_hparams(best_config, parameters['hparam_map']))
    ray.shutdown()

if __name__ == '__main__':
    main()
