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
""" Implements the Strategist interface. """

import abc
import numpy as np


class Strategist(abc.ABC):
    """ Implements the Strategist interface.

    The Strategist class is responsible for producing actions inside a
    blackjack game such as splitting, doubling down, hitting and standing
    based on the hand of the player, dealer and the distribution of cards
    remaining in the shoe.
    """

    @abc.abstractmethod
    def should_split(self,
                     player_total: int,
                     player_aces: int,
                     dealer_total: int,
                     card_distribution: np.ndarray) -> bool:
        """ Returns whether it is optimal to split in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        pass

    @abc.abstractmethod
    def should_double(self,
                      player_total: int,
                      player_aces: int,
                      dealer_total: int,
                      card_distribution: np.ndarray) -> bool:
        """ Returns whether it is optimal to double down in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        pass

    @abc.abstractmethod
    def should_hit(self,
                   player_total: int,
                   player_aces: int,
                   dealer_total: int,
                   card_distribution: np.ndarray) -> bool:
        """ Returns whether it is optimal to hit in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        pass

    @abc.abstractmethod
    def free_mem(self) -> None:
        pass
