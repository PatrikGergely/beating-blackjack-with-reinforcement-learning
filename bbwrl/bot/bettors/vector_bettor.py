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
""" Implements a vector bettor. """

from bbwrl.bot.bettors.bettor import Bettor
import numpy as np


class VectorBettor(Bettor):
    """ Implements a vector bettor.

    Implements a bettor that bets based on the dot product of the remaining
    cards in the shoe and a special vector.

    Attributes:
        _shoe_size: The number of decks in the shoe.
        _vector: The special vector used in the dot product.
    """
    def __init__(self, shoe_size: int, vector: np.ndarray):
        """ Initializes the vector bettor.

        Args:
            shoe_size: The number of decks in the shoe.
            vector: The special vector used in the dot product.
        """
        self._shoe_size = shoe_size
        self._vector = np.array(vector)

    def _running_count(self, card_distribution: np.ndarray) -> float:
        """ Calculates the running count. """
        count = 0.0
        for i in range(1, 14):
            count += self._vector[i] * (self._shoe_size*4 -
                                        card_distribution[i])
        return count

    def _true_count(self, card_distribution: np.ndarray) -> float:
        """ Calculates the true count. """
        decks_left = card_distribution.sum() / 52.0
        return self._running_count(card_distribution) / decks_left

    def get_bet_size(self,
                     chips: float,
                     card_distribution: np.ndarray) -> float:
        """ Returns the bet size based on the true count.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        return self._true_count(card_distribution) - 1.0

    def set_payout(self, payout: float, card_distribution: np.ndarray) -> None:
        return

    def save(self) -> None:
        return
