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
""" Implements the TrainerBettor class"""

import math
from typing import Any, Callable

from acme import types
from acme.agents.agent import Agent
from dm_env import TimeStep, StepType
import numpy as np

from bbwrl.bot.bettors.bettor import Bettor
from bbwrl.environments import rule_variation


MIN_REWARD = -1e60


def _log_util(x: float) -> float:
    """ Calculates the logarithm utility. """
    if x+1 <= 0:
        return MIN_REWARD
    if math.log(x+1) <= MIN_REWARD:
        return MIN_REWARD
    return math.log(x+1)


class TrainerBettor(Bettor):
    """ Implements a bettor that trains an ACME Agent.

    Attributes:
        _agent: The agent to train.
        _action2bet: A function that maps the agent action to a bet size.
        _dtype: The dtype used by the agent.
        _utility_function: The utility function used by the agent.
        _last_action: The last action taken by the bettor.
        _chips_before_game: The bankroll of the player before the last bet.
    """

    def __init__(self,
                 agent: Agent,
                 action2bet: Callable[[Any], float],
                 dtype: str,
                 utility_function: Callable[[float], float] = _log_util):
        """ Initializes the TrainerBettor.

        Args:
            agent: The agent to train.
            action2bet: A function that maps the agent action to a bet size.
            dtype: The dtype used by the agent.
            utility_function: The utility function used by the agent.
        """
        self._last_action = None
        self._utility_function = utility_function
        self._agent = agent
        self._action2bet = action2bet
        self._dtype = dtype
        self._chips_before_game = rule_variation.AGENT_CHIPS

    def _default_spec(self,
                      chips: float,
                      card_distribution: np.ndarray) -> TimeStep:
        """ Returns the initial TimeStep.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        return TimeStep(step_type=StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=self._create_observation(chips,
                                                        card_distribution))

    def _create_timestep(self,
                         chips: float,
                         reward: np.ndarray,
                         card_distribution: np.ndarray) -> TimeStep:
        """ Returns a TimeStep.

        Args:
            chips: The bankroll of the player.
            reward: The reward for the last action.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        if chips < 0:
            st = StepType.LAST
        else:
            st = StepType.MID
        return TimeStep(step_type=st,
                        reward=reward,
                        discount=np.array(1.0, dtype=self._dtype),
                        observation=self._create_observation(chips,
                                                        card_distribution))

    def _create_observation(self,
                            chips: float,
                            card_distribution: np.ndarray) -> types.NestedArray:
        """ Creates an observation.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        return {
            'CHIPS': np.array([chips], dtype=self._dtype),
            'CARD_DISTRIBUTION': card_distribution.astype(self._dtype)
        }

    def get_bet_size(self,
                     chips: float,
                     card_distribution: np.ndarray) -> float:
        """ Returns a bet size.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        if self._last_action is None:
            self._agent.observe_first(self._default_spec(chips,
                                                    card_distribution))
        self._last_action = self._agent.select_action(
            self._create_observation(chips, card_distribution))
        # The action sometimes is a scalar sometimes an array
        self._last_action = self._last_action.flatten()[0]
        self._chips_before_game = chips
        return self._action2bet(self._last_action)

    def _get_reward(self, payout: float) -> np.ndarray:
        """ Returns reward corresponding to payout

        Args:
            payout: The payout to find the corresponding reward to.
        """
        reward = (self._utility_function(self._chips_before_game+payout)
                  - self._utility_function(self._chips_before_game))
        return np.array(reward, dtype=self._dtype)

    def set_payout(self,
                   payout: float,
                   card_distribution: np.ndarray) -> None:
        """ Update the agent based on the payout.

        Calculate the reward corresponding to the payout and update the agent.

        Args:
            payout: The payout of the last action.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        reward = self._get_reward(payout)
        timestep = self._create_timestep(self._chips_before_game + payout,
                                         reward,
                                         card_distribution)
        self._agent.observe(self._last_action, timestep)
        self._agent.update()
        if timestep.last():
            self._last_action = None

    def save(self) -> str:
        """ Saves the network corresponding to the agent's learner.

        Returns:
            The path to the saved network.
        """
        # pylint: disable = protected-access
        self._agent._learner._snapshotter.save(True)
        return self._agent._learner._snapshotter.directory
        # pylint: enable = protected-access
