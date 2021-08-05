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
""" Evaluates a bot by approximating E[log(1+x)].

Use: python bbwrl/evaluator.py simulation/*
"""

import math
import os
import sys
from typing import TextIO, Tuple, List, Dict, Any

import numpy as np
from matplotlib import pyplot as plt

from bbwrl import simulator


def _get_line(reader: TextIO) -> str:
    """ Returns the next line from a file.

    Args:
        reader: The reader that reads from the file.

    Returns:
        The next line that corresponds to an action taken by the bot.
    """
    while True:
        line = reader.readline()
        if line in ['', 'START\n'] or line.split(',')[0] in ['CHOOSE_BET',
                                                             'SPLIT?',
                                                             'DOUBLE?',
                                                             'HIT/STAND',]:
            return line


def _read_file(file_name: str) -> Tuple[List[List[float]],
                                        List[List[float]],
                                        List[List[float]],
                                        List[List[float]]]:
    """ Reads in a simulation file.

    Args:
        file_name: The name of the file containing the simulation.

    Returns:
        Four lists that contain a snapshot after each bet of the bankroll,
        the bet size last produced, the payout last obtained and the geometric
        increase of the bankroll.
    """
    with open(file_name, 'r') as reader:
        line = _get_line(reader)
        chips = []
        rewards = []
        bets = []
        geom_incs = []
        while line != '':
            tmp_chips = []
            tmp_rewards = []
            tmp_bets = []
            tmp_geom_incs = []
            while line != '' and line[:5] != 'START':
                (move_type, chip, _, _,
                 _, action, _) = line.split(',')
                if move_type == 'CHOOSE_BET':
                    tmp_chips.append(float(chip))
                    if len(tmp_chips) > 1:
                        reward = (tmp_chips[-1]-tmp_chips[-2])/tmp_bets[-1]
                        tmp_rewards.append(round(2*reward)/2)
                        tmp_geom_incs.append((1+tmp_chips[-1])/
                                             (1+tmp_chips[-2])-1)
                    else:
                        tmp_rewards.append(0.0)
                        tmp_geom_incs.append(0.0)
                    tmp_bets.append(float(action))
                line = _get_line(reader)
            if len(tmp_chips) > 0:
                chips.append(tmp_chips)
                bets.append(tmp_bets)
                rewards.append(tmp_rewards)
                geom_incs.append(tmp_geom_incs)
            line = _get_line(reader)
    return chips, bets, rewards, geom_incs


def log_metric(geom_inc: List[List[float]]) -> float:
    """ Returns the approximation of E[log(1+x)].

    Args:
        geom_inc: The geometric increases corresponding to a simulation.
    """
    geom_inc = np.concatenate(geom_inc).flat
    mean = np.mean(geom_inc)
    var = np.var(geom_inc)
    return math.log(1+mean) - (var / (2*(1+mean)*(1+mean)))


def read_logs(file_names: List[str]) -> Tuple[List[List[float]],
                                              List[List[float]],
                                              List[List[float]],
                                              List[List[float]]]:
    """ Reads multiple simulation files.

    Args:
        file_names: The name of the files containing the simulations.

    Returns:
        Four lists that contain a snapshot after each bet of the bankroll,
        the bet size last produced, the payout last obtained and the geometric
        increase of the bankroll.
    """
    chips = []
    bets = []
    rewards = []
    geom_incs = []
    for file_name in file_names:
        tmp_chips, tmp_bets, tmp_rewards, tmp_geom_incs = _read_file(file_name)
        chips += tmp_chips
        bets += tmp_bets
        rewards += tmp_rewards
        geom_incs += tmp_geom_incs
    return chips, bets, rewards, geom_incs


def evaluate_simulations(file_names: List[str],
                         output_path: str = None) -> float:
    """ Evaluate simulations after reading them from a file.

    Args:
        file_names: The name of the files containing the simulations.

    Returns:
        The approximation of E[log(1+x)] based on the simulations.
    """
    chips, bets, _, geom_incs = read_logs(file_names)
    metric = log_metric(geom_incs)
    if output_path is not None:
        _plot(chips, bets, metric, output_path)
    return metric


def evaluate(bettor_name: str,
             bettor_params: Dict[str, Any],
             strategist_name: str,
             max_episode_length: int,
             num_episodes: int,
             simulation_id: str,
             path: str = None) -> float:
    """ Evaluates a bot after simulating it for a required number of steps.

    Args:
        bettor_name: The name of the bettor to use.
        bettor_params: The parameter of the bettor to initialize with.
        strategist_name: The name of the strategist to use.
        max_episode_length: The number of games to terminate the episode after.
        num_episodes: The number of episodes to simulate the bot for.
        simulation_id: The of the simulation to avoid overwriting files.
        path: The path to save the simulation to. If not given, the simulation
            is placed in the current directory and deleted afterwards.
    """
    auto_delete = path is None
    if auto_delete:
        path = '.'
    log = simulator.simulate(bettor_name,
                             bettor_params,
                             strategist_name,
                             max_episode_length,
                             num_episodes,
                             simulation_id,
                             path)
    metric = evaluate_simulations([log])
    if auto_delete:
        os.remove(log)
    return metric


def _fill_arrays(arrays: List[List[float]]) -> List[List[float]]:
    maxn = 0
    for array in arrays:
        maxn = max(maxn, len(array))
    modified_arrays=[]
    for array in arrays:
        modified_arrays.append(array + [0.0]*(maxn-len(array)))
    return modified_arrays


def _plot(chips: List[List[float]],
          bets: List[List[float]],
          metric: float,
          path: str) -> None:
    fig, axs = plt.subplots(2, figsize=(6,12))
    fig.suptitle('{}'.format(metric))
    modified_chips = _fill_arrays(chips)
    modified_bets = _fill_arrays(bets)
    axs[0].plot(np.transpose(modified_chips))
    axs[1].plot(np.transpose(modified_bets))
    plt.savefig(path)


def main():
    """ Reads in simulation files and approximates E[log(1+x)] for the bot.

    The function is capable of reading in different bots and group them by
    their simulation type.
    """
    simulations_by_types = {}
    for filename in sys.argv[1:]:
        #simulation_type = filename[:filename.rfind('_')]
        simulation_type = '_'.join(filename.split('_')[:-2])
        simulation_type = simulation_type[simulation_type.rfind('/')+1:]
        if simulation_type in simulations_by_types:
            simulations_by_types[simulation_type].append(filename)
        else:
            simulations_by_types[simulation_type]=[filename]
    for simulation_type in simulations_by_types:
        filenames = simulations_by_types[simulation_type]
        output = simulation_type + '.pdf'
        metric = evaluate_simulations(filenames, output)
        print('The {} many files starting with {} has a metric of {}'.format(
            len(filenames),
            simulation_type,
            metric))


if __name__ == '__main__':
    main()
