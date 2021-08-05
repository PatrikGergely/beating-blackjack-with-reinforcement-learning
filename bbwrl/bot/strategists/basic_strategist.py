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
""" Implements a basic strategist.

The strategist uses a lookup table to decide what actions to take based on
the hand of the dealer and the player while disregarding the distribution of
the card remaining in the deck.

The lookup tables were generated with:
bbwrl.utils.basic_strategy_generator.main()
"""
import numpy as np

from bbwrl.bot.strategists.strategist import Strategist


BASIC_ACE_STRATEGY = [
    # 2    3    4    5    6    7    8    9    10   A
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 12
    ['H', 'H', 'H', 'H', 'D', 'H', 'H', 'H', 'H', 'H'], # 13
    ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'], # 14
    ['H', 'H', 'H', 'D', 'D', 'H', 'H', 'H', 'H', 'H'], # 15
    ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'], # 16
    ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'], # 17
    ['S', 'D', 'D', 'D', 'D', 'S', 'S', 'H', 'H', 'H'], # 18
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 19
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 20
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 21
]


BASIC_HIT_STRATEGY = [
    # 2    3    4    5    6    7    8    9    10   A
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 3
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 4
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 5
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 6
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 7
    ['H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H'], # 8
    ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'], # 9
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'], # 10
    ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H'], # 11
    ['H', 'H', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'], # 12
    ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'], # 13
    ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'], # 14
    ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'], # 15
    ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'S', 'H'], # 16
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 17
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 18
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 19
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 20
    ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], # 21
]


BASIC_SPLIT_STRATEGY = [
    #  2      3      4      5      6      7      8      9      10     A
    [True , True , True , True , True , True , True , True , True , True ], # As
    [True , True , True , True , True , True , False, False, False, False], # 4
    [True , True , True , True , True , True , False, False, False, False], # 6
    [False, False, False, True , True , False, False, False, False, False], # 8
    [False, False, False, False, False, False, False, False, False, False], # 10
    [True , True , True , True , True , False, False, False, False, False], # 12
    [True , True , True , True , True , True , False, False, False, False], # 14
    [True , True , True , True , True , True , True , True , True , True ], # 16
    [True , True , True , True , True , False, True , True , False, False], # 18
    [False, False, False, False, False, False, False, False, False, False], # 20
]


class BasicStrategist(Strategist):
    """ Implements a basic strategist using a lookup table to choose actions.
    """
    def __init__(self, utility_function):
        pass

    def _prefered_move(self,
                       player_total: int,
                       player_aces: int,
                       dealer_total: int) -> str:
        """Decides whether the player should double down, hit or stand.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of aces which have the possibility to be 1
                or 11 of the player.
            dealer_total: The hand total of the dealer.

        Returns:
            The prefered move among doubling down, hitting and standing.
        """
        if player_aces == 1:
            return BASIC_ACE_STRATEGY[player_total-12][dealer_total-2]
        return BASIC_HIT_STRATEGY[player_total-3][dealer_total-2]

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
        if player_aces == 1:
            return BASIC_SPLIT_STRATEGY[0][dealer_total-2]
        return BASIC_SPLIT_STRATEGY[int(player_total/2)-1][dealer_total-2]

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
        return self._prefered_move(player_total,
                                   player_aces,
                                   dealer_total) == 'D'

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
        return self._prefered_move(player_total,
                                   player_aces,
                                   dealer_total) != 'S'

    def free_mem(self) -> None:
        pass
