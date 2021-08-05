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
""" Trains a ML Bettor.

Use: python bbwrl/trainer.py <train_config_file> <output_file>

Trains and evaluates a ML bettor and prints result and the model location.

Output file is optional, when present output is appended to the file, otherwise
output is printed to the standard output.

Example config file:
{
    "agent_name": "DDPG",
    "strategist_name": "BasicStrategist",
    "hparams":
    {
        "policy_network_shape": [64],
        "critic_network_shape": [4, 8],
        "sigma": 0.257,
        "policy_learning_rate": 2e-18,
        "critic_learning_rate": 5e-14,
        "batch_size": 16
    },
    "time_limit": 36000,
    "max_episode_length": 10000,
    "train_num_episodes": 20,
    "eval_num_episodes": 50
}
"""

import json
import sys
import time
from typing import Any, Dict, Tuple
import uuid
import os

import acme
import acme.utils.paths

from bbwrl.bot import Bot
from bbwrl.environments import Table
from bbwrl import evaluator


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(bettor_name: str,
          strategist_name: str,
          hparams: Dict[str, Any],
          time_limit: int,
          max_episode_length: int,
          num_episodes: int) -> str:
    """ Trains an ML based bettor.

    Args:
        bettor_name: The name of the bettor to train (DQNBettor or DDPGBettor).
        strategist_name: The name of the strategist to use.
        hparams: The hyper parameters to initalize the trainable bettor with.
        time_limit: The number of seconds to train the bettor for.
        max_episode_length: The number of games to terminate the episode after.
        num_episodes: The number of episodes to simulate the bot for.

    Returns:
        The location of the trained bettor.
    """
    # When training multiple bettors on the same process (for example during
    # tuning) the _ACME_ID needs to be regenerated to avoid overwriting
    # previous models.
    acme.utils.paths._ACME_ID = uuid.uuid1()  # pylint: disable=protected-access
    bot = Bot(bettor_name,
              hparams,
              strategist_name)
    env_loop = acme.EnvironmentLoop(Table(max_episode_length), bot)
    start_time = time.time()
    while (time.time() - start_time) < time_limit:
        env_loop.run(num_episodes=num_episodes)
    return bot.save()


def train_and_evaluate(agent_name: str,
                       strategist_name: str,
                       hparams: Dict[str, Any],
                       time_limit: int,
                       max_episode_length: int,
                       train_num_episodes: int,
                       eval_num_episodes: int) -> Tuple[str, float]:
    """ Train and then evaluate a bettor.

    Args:
        agent_name: The name of the type of agent to train (DQN or DDPG).
        strategist_name: The name of the strategist to use.
        hparams: The hyper parameters to initalize the trainable bettor with.
        time_limit: The number of seconds to train the bettor for.
        max_episode_length: The number of games to terminate the episode after.
        train_num_episodes: The number of episodes to simulate the bot for at a
            time when training the bettor.
        eval_num_episodes: The number of episodes to simulate the bot for when
            evaluating.

    Args:
        The location of the trained bettor and the score obtained during
        evaluation.
    """
    network_path = train(agent_name+'Trainer', strategist_name, hparams,
                         time_limit, max_episode_length, train_num_episodes)
    metric = evaluator.evaluate(agent_name+'Bettor',
                                {'policy_path': network_path+'/network'},
                                strategist_name,
                                max_episode_length,
                                eval_num_episodes,
                                network_path.split('/')[-2])
    return network_path, metric


def main():
    with open(sys.argv[1]) as json_file:
        data = json.load(json_file)
    network_path, metric = train_and_evaluate(**data)
    if len(sys.argv) > 2:
        with open(sys.argv[2], 'a') as output:
            # Append 'hello' at the end of file
            output.write('Metric: {}, Network path: {}\n'.format(metric,
                                                               network_path))
    else:
        print('The trained network is saved at location '+
              '{} and evaluated with metric {}'.format(network_path, metric))

if __name__ == '__main__':
    main()
