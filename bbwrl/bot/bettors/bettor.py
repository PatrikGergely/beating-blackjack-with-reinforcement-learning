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
""" Implements the Bettor interface. """

import abc
from typing import Optional

import numpy as np


class Bettor(abc.ABC):
    """ Implements the Bettor interface.

    The Bettor class is responsible for producing bet sizes based on the
    chips of the player and the card distribution of the remaining cards of
    the shoe.
    """

    @abc.abstractmethod
    def get_bet_size(self,
                     chips: float,
                     card_distribution: np.ndarray) -> float:
        """ Returns a bet size.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        pass

    @abc.abstractmethod
    def set_payout(self, payout: float, card_distribution: np.ndarray) -> None:
        """ Update the Bettor when necessary.

        Args:
            payout: The payout of the last action.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        pass

    @abc.abstractmethod
    def save(self) -> Optional[str]:
        """ Saves the model when necessary.

        Returns:
            The path to the saved bettor if saved, otherwise, None.
        """
        pass
