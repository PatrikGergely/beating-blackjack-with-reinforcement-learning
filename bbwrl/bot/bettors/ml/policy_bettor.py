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
""" Implements the PolicyBettor class"""

from typing import Any, Callable

from acme import types
import numpy as np
import tensorflow as tf

from bbwrl.bot.bettors.bettor import Bettor


class PolicyBettor(Bettor):
    """ Implements a bettor that uses a saved tensorflow model.

    After training a Bettor using TrainerBettor, the saved model is loaded
    using this class to follow the policy given by the saved model.
    """
    def __init__(self,
                 policy_path: str,
                 policy2action: Callable[[Any], float],
                 obs_precomp: Callable[[types.NestedArray], Any],
                 dtype: str):
        """ Initializes the bettor.

        Args:
            policy_path: Path to the saved model.
            policy2action: A function that maps the policy output to an action.
            obs_precomp: A function that maps the observations to the required
                input for the policy network.
            dtype: The dtype used by the network.
        """
        self._loaded_policy = tf.saved_model.load(policy_path)
        self._policy2action = policy2action
        self._obs_precomp = obs_precomp
        self._dtype = dtype

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
            'CHIPS':
                np.array([[chips]], dtype=self._dtype),
            'CARD_DISTRIBUTION':
                card_distribution.astype(self._dtype).reshape(1, 14)
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
        obs = self._obs_precomp(self._create_observation(chips,
                                                         card_distribution))
        return self._policy2action(self._loaded_policy(obs))

    def set_payout(self, payout: float, card_distribution: np.ndarray) -> None:
        pass

    def save(self) -> None:
        pass
