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
""" Simulates a bot.

Use: python bbwrl/simulator.py <simulation_config_file> <process_id>

Example config file:
{
    "bettor_name": "ConstantBettor",
    "bettor_params": {},
    "strategist_name": "BasicStrategist",
    "max_episode_length": 10000,
    "num_episodes": 10,
    "simulation_id": 1,
    "path": "./simulations"
}
"""

import json
import sys
from typing import Any, Dict

import acme

from bbwrl.bot import Bot
from bbwrl.environments import Table
from bbwrl.utils import logger


def simulate(bettor_name: str,
             bettor_params: Dict[str, Any],
             strategist_name: str,
             max_episode_length: int,
             num_episodes: int,
             simulation_id: str,
             path: str,
             process_id: int = 0) -> str:
    """ Simulates a bot and logs the actions to a file.

    Args:
        bettor_name: The name of the bettor to use.
        bettor_params: The parameter of the bettor to initialize with.
        strategist_name: The name of the strategist to use.
        max_episode_length: The number of games to terminate the episode after.
        num_episodes: The number of episodes to simulate the bot for.
        simulation_id: The of the simulation to avoid overwriting files.
        path: The path to save the simulation to. If not given, the simulation
            is placed in the current directory and deleted afterwards.
        process_id: An optional argument to allow the same cpu log into
            different files on each process.

    Returns:
        The filename of the simulation.
    """
    filename = '{}/{}_{}_{}_{}.csv'.format(path,
                                           bettor_name[:-6].upper(),
                                           strategist_name[:-10].upper(),
                                           simulation_id,
                                           process_id)
    env_loop = acme.EnvironmentLoop(Table(max_episode_length),
                                    Bot(bettor_name,
                                        bettor_params,
                                        strategist_name,
                                        logger.create_logger(filename)))
    env_loop.run(num_episodes=num_episodes)
    return filename


def main():
    with open(sys.argv[1]) as json_file:
        data = json.load(json_file)
    if len(sys.argv) > 2:
        process_id = sys.argv[2]
    else:
        process_id = 1
    simulate(process_id=process_id, **data)


if __name__ == '__main__':
    main()
