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

"""Blackjack playing bot implementation"""

import logging
import math
from typing import Any, Dict

import acme.core
from acme import types
import dm_env
import numpy as np

from bbwrl.bot import bettors
from bbwrl.bot import strategists
from bbwrl.environments import rule_variation


# The minimum receivable reward
MIN_REWARD = -1e60


def _create_action(value: float) -> types.NestedArray:
    """ Returns float64 numpy scalar.

    Args:
        value: The value to convert.
    """
    return np.array(value, dtype=np.float64)


def _log_util(chips: float,
              bet_size: float,
              payout: float) -> float:
    """ Returns the gained log utility.

    Returns the difference between the utility before and after a payout.
    The payout is atleast MIN_REWARD.

    Args:
        chips: The current number of chips the agent owns.
        bet_size: The size of the bet the agent takes.
        payout: The payout of the game.
    """
    if chips <= 0 or chips + payout*bet_size <= 0:
        return MIN_REWARD
    return max(math.log(1.0 + chips + payout*bet_size) - math.log(1.0 + chips),
               MIN_REWARD)

class Bot(acme.core.Actor):
    """ Implements an actor capable of interacting with Table.

    The actor takes observations from a Table object and after delegating
    computation to either it's strategist or bettor, returns an action to the
    environment.

    Attributes:
        _bettor: The bettor that determines bet sizes.
        _strategist_class: The class of the strategist to be used.
        _strategist: The strategist used to determine what action to take.
        _logger: The logger used to log every action-observation pair.
        _last_timestep: The timestep before the last action was taken.
    """
    def __init__(self,
                 bettor_name: str,
                 bettor_parameters: Dict[str, Any],
                 strategist_name: str,
                 logger: logging.Logger = None):
        """ Initializes the bot.

        Args:
            bettor_name: The name of the bettor to initialize.
            bettor_parameters: The parameters to initialize the bettor with.
            strategist_name: The name of the strategist class to use.
            logger: The logger to use to log every action-observation pair.
        """
        self._bettor = bettors.BETTORS[bettor_name](**bettor_parameters)
        self._strategist_class = strategists.STRATEGISTS[strategist_name]
        self._strategist = None
        self._logger = logger
        self._reset_deck()

    def select_action(self,
                      observation: types.NestedArray) -> types.NestedArray:
        """ Selects an action to take.

        Args:
            observation: The last observation of the environment

        Returns:
            A nested array holding the chosen action.
        """
        if observation['STAGE'] == 'CHOOSE_BET':
            chips = observation['CHIPS']
            bet_size = self._bettor.get_bet_size(chips, self._deck_distribution)
            if np.isnan(bet_size):
                bet_size = 1.0
            bet_size = min(1000.0, chips, bet_size)
            bet_size = max(1.0, bet_size)
            if self._strategist is not None:
                self._strategist.free_mem()
            self._strategist = self._strategist_class(
                lambda x: _log_util(chips, bet_size, x))
            return _create_action(bet_size)
        player_total = observation['PLAYER_TOTAL']
        player_aces = observation['PLAYER_ACES']
        dealer_total = observation['DEALER_TOTAL']
        if observation['STAGE'] == 'SPLIT?':
            split = self._strategist.should_split(player_total,
                                                  player_aces,
                                                  dealer_total,
                                                  self._deck_distribution)
            return _create_action(split)
        if observation['STAGE'] == 'DOUBLE?':
            double = self._strategist.should_double(player_total,
                                                    player_aces,
                                                    dealer_total,
                                                    self._deck_distribution)
            return _create_action(double)
        if observation['STAGE'] == 'HIT/STAND':
            hit = self._strategist.should_hit(player_total,
                                              player_aces,
                                              dealer_total,
                                              self._deck_distribution)
            return _create_action(hit)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        """ Observes the first timestep.

        Args:
            timestep: The first timestep
        """
        self._last_timestep = timestep
        if self._logger is not None:
            self._logger.info('START')
        self._reset_deck()

    def _log(self, action: types.NestedArray) -> None:
        """ Logs the current action-observation pair.

        Logs the last observation, the action taken afterwards and the
        current deck distribution.

        Args:
            action: The action last taken.
        """
        if self._logger is None:
            return
        self._logger.info('{}, {}, {}, {}, {}, {}, {}'.format(
            self._last_timestep.observation['STAGE'],
            self._last_timestep.observation['CHIPS'],
            self._last_timestep.observation['PLAYER_TOTAL'],
            self._last_timestep.observation['PLAYER_ACES'],
            self._last_timestep.observation['DEALER_TOTAL'],
            action,
            self._deck_distribution))

    def observe(self,
                action: types.NestedArray,
                next_timestep: dm_env.TimeStep) -> None:
        """ Observes the next timestep after performing an action.

        Logs the last action-observation pair and updates the deck
        distribution.

        Args:
            action: The last action taken.
            next_timestep: The timestep after the performed action.
        """
        self._log(action)
        self._last_timestep = next_timestep
        if next_timestep.observation['REVEALED_CARDS'][0] == -1:
            self._reset_deck()
        self._deck_distribution -= next_timestep.observation['REVEALED_CARDS']
        self._deck_distribution[0] = 0
        if next_timestep.observation['STAGE'] == 'CHOOSE_BET':
            self._bettor.set_payout(next_timestep.reward,
                                    self._deck_distribution)

    def update(self) -> None:
        pass

    def _reset_deck(self) -> None:
        """ Resets the deck distribution when reshuffled. """
        self._deck_distribution = np.full(14,
                                          4*rule_variation.SHOE_SIZE)
        self._deck_distribution[0] = 0

    def save(self) -> str:
        """ Save the neural network of the bettor.

        Returns:
            The path where the neural network is stored.
        """
        return self._bettor.save()
