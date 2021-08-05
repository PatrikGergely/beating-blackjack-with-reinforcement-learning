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

"""Blackjack shoe implementation."""
from typing import List

import numpy as np

from bbwrl.environments import rule_variation


# Type alias to differentiate between cards and their values.
# The card 1 corresponds to an Ace, which values 11.
# Cards over 10 correspond to face cards, which value 10.
Card = int


class Shoe(object):
    """ Represents a blackjack shoe.

    A shoe contains multiple decks, that the environment can draw from and
    when only a small percentage of cards are left it automatically reshuffles.

    Attributes:
        _full_shoe: A list cards that make the shoe before shuffleing.
        _it: The iterator of the current card in the deck.
        _rng: The random number generator used by the shoe.
        _running_shoe: A list of cards that represent the order of cards in a
            shuffled shoe.
    """

    def __init__(self):
        """ Initializes a shoe object.

        Sets up the full shoe based on the number of shoes required by the
        rule variation, saves a random number generator and shuffles the shoe.
        """
        self._full_shoe = np.repeat(np.arange(1, 14),
                                    4*rule_variation.SHOE_SIZE)
        self._rng = np.random.default_rng()
        self.reshuffle()

    def cards_left(self) -> float:
        """ Returns the percentage of cards that have not yet been drawn. """
        return 1 - (self._it / np.size(self._full_shoe))

    def draw(self) -> Card:
        """ Returns the next card in the deck. """
        self._it += 1
        return self._running_shoe[self._it - 1]

    def reshuffle(self) -> None:
        """ Reshuffles the shoe. """
        self._running_shoe = self._full_shoe.copy()
        self._it = 0
        self._rng.shuffle(self._running_shoe)

    def try_reshuffle(self) -> bool:
        """ Reshuffles if there are less cards left than a threshold. """
        if self.cards_left() < rule_variation.RESHUFFLE:
            self.reshuffle()
            return True
        return False


# pylint: disable=protected-access
def _custom_shoe(cards: List[Card]) -> Shoe:
    """ Returns a shoe where the top cards are given.

    Args:
        cards: The list of cards that will be on the top of the deck. The first
            card in cards will be the first card to be drawn from the deck.
    """
    shoe = Shoe()
    running_shoe = shoe._running_shoe
    for card in cards:
        position = np.where(running_shoe == card)[0][0]
        running_shoe = np.delete(running_shoe, position)
    shoe._rng.shuffle(running_shoe)
    shoe._running_shoe = np.append(cards, running_shoe)
    return shoe
