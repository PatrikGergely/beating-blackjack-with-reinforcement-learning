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

"""Blackjack game implementation."""

import numpy as np
from typing import Dict, Optional

from bbwrl.environments.shoe import Shoe, Card
from bbwrl.environments import rule_variation


def _get_payout(player: int, dealer: int) -> float:
    """Calculates the payout for the given hand totals.

    Args:
        player: The hand total of the player.
        dealer: The hand total of the dealer.

    Returns:
        The payout of the payout.
    """
    if player > 21:
        return -1.
    if dealer > 21:
        return 1.
    if player == dealer:
        return 0.
    return 1. if player > dealer else -1.


def value(card: Card) -> int:
    """Gets the value of a given card.

    Number cards count as their natural value; face cards count as 10;
    aces are valued as 11

    Args:
        card: The card in question.

    Return:
        The value of the given card.
    """
    if card == 1:
        return 11
    if card > 10:
        return 10
    return card


class Dealer:
    """Represents a blackjack dealer.

    Dealer is initially dealt two cards, one of which is shown to the player.
    When the player stands the dealer follows a fixed set of rules to determine
    its final hand total.

    Attributes:
        _aces: The number of aces which have the possibility to be 1 or 11.
        _blackjack: Whether the dealer was dealt blackjack.
        _hidden_card: The initial card which is not shown to the player.
        _total: The current hand total.
    """
    def __init__(self,
                 shown_card: Card,
                 hidden_card: Card):
        """Initializes the dealer using two dealt cards.

        Args:
            shown_card: The initial card which is shown to the player.
            hidden_card: The initial card which is not shown to the player.
        """
        self._total = 0
        self._aces = 0
        self._add_card(shown_card)
        self._blackjack = self._total + value(hidden_card) == 21
        self._hidden_card = hidden_card

    def _add_card(self, card: Card) -> None:
        """Updates the hand total with a card.

        Adds the value of a card to the hand total. In case of a soft hand
        exceeding 21, the hand becomes hard and gets substracted 10 points.

        Args:
            card: The card to be added to the hand.
        """
        self._total += value(card)
        self._aces += 1 if (card == 1) else 0
        if self._total > 21:
            if self._aces > 0:
                self._aces -= 1
                self._total -= 10

    def get_blackjack(self) -> bool:
        return self._blackjack

    def get_aces(self) -> int:
        return self._aces

    def get_total(self) -> int:
        return self._total

    def _needs_to_draw(self) -> bool:
        """Decides whether the dealer needs to draw an additional card.

        Dealers always hit under 17 and stand otherwise. Whether the dealer
        hits soft 17 depends on the rule variation.

        Returns:
            A boolean whether the dealer needs to draw an additional card.
        """
        if self._total == 17:
            return rule_variation.HIT_SOFT_17 and self._aces > 0
        return self._total < 17

    def stand(self, shoe: Shoe) -> np.ndarray:
        """Draws cards until required.

        Draws additional cards until it is required by the rules of blackjack.

        Args:
            shoe: The shoe where the card is drawn from

        Returns:
            A numpy array with shape (14, ) where each entry represents how
            many of the corresponding cards were drawn.
        """
        currently_shown = np.zeros(14, dtype=int)
        card = self._hidden_card
        self._add_card(card)
        currently_shown[card] += 1
        while self._needs_to_draw():
            card = shoe.draw()
            self._add_card(card)
            currently_shown[card] += 1
        return currently_shown


class Player:
    """Represents a blackjack player.

    Players are initially dealt two cards. If these are identical, the player
    has the option to split them into two different hands. Before hitting for
    the first time, the player has the option to double down, when the player
    is dealt exactly one card and recieves double payout. The player can hit
    to recieve an additional card until they stand.

    Attributes:
        _aces: The number of aces which have the possibility to be 1 or 11.
        _blackjack: Whether the player won with blackjack.
        _can_split: Whether the player is allowed to split.
        _can_double_down: Whether the player is allowed to double down.
        _doubled_down: Whether the player doubled down.
        _stand: Whether the player is standing.
        _total: The current hand total.
    """

    def __init__(self,
                 first_card: Card,
                 second_card: Card,
                 can_double_down: bool = True,
                 can_blackjack: bool = True):
        """Initializes the player using two dealt cards.

        Args:
            first_card: The first initial card.
            second_card: The second initial card.
            can_double_down: Whether the player is allowed to double down.
            can_blackjack: Whether the player is rewarded extra for winning
                with blackjack.
        """
        self._total = 0
        self._aces = 0
        self._add_card(first_card)
        self._add_card(second_card)
        self._can_split = (value(first_card) == value(second_card)
                           if rule_variation.SPLIT_UNEVEN
                           else first_card == second_card)
        self._doubled_down = False
        self._blackjack = self.get_total() == 21 and can_blackjack
        self._stand = self.get_total() == 21
        self._can_double_down = can_double_down and not self._stand

    def _add_card(self, card: Card) -> None:
        """Updates the hand total with a card.

        Adds the value of a card to the hand total. In case of a soft hand
        exceeding 21, the hand becomes hard and gets substracted 10 points.
        If a hard hand exceeds 21, the player busts and automtically stands.

        Args:
            card: The card to be added to the hand.
        """
        self._total += value(card)
        self._aces += 1 if (card == 1) else 0
        if self._total > 21:
            if self._aces > 0:
                self._aces -= 1
                self._total -= 10
        if self._total >= 21:
            self._stand = True

    def double_down(self, shoe: Shoe) -> Optional[Card]:
        """Doubles down.

        Stands after drawing a single card. The payout for this player will be
        doubled.

        Args:
            shoe: The shoe where the card is drawn from

        Returns:
            None if the player was not allowed to double down, otherwise the
            drawn card.
        """
        if not self._can_double_down:
            return
        self._doubled_down = True
        card = self.hit(shoe)
        self._stand = True
        return card

    def get_aces(self) -> int:
        return self._aces

    def get_blackjack(self) -> bool:
        return self._blackjack

    def get_doubled_down(self) -> bool:
        return self._doubled_down

    def get_stand(self) -> bool:
        return self._stand

    def get_total(self) -> int:
        return self._total

    def hit(self, shoe: Shoe) -> Optional[Card]:
        """ Draws a single card if haven't stood yet.

        Draws a single card and adds its value to the hand total. Splitting and
        doubling down is not allowed afterwards.

        Args:
            shoe: The shoe where the card is drawn from

        Returns:
            None if the player was not allowed to hit, otherwise the drawn
            card.
        """
        if self._stand:
            return
        self._can_split = False
        self._can_double_down = False
        card = shoe.draw()
        self._add_card(card)
        return card

    def split_value(self) -> Optional[Card]:
        """Returns the initial card if the hand can be split.

        Returns:
            The initial card if the hand can be split otherwise returns None.
        """
        if not self._can_split:
            return None
        if self._aces != 0:
            return 1
        return int(self._total / 2)

    def stand(self) -> None:
        self._stand = True


class Game:
    """ Implements a blackjack game.

    Holds a shoe, dealer and players and performs actions such as hitting and
    standing on the appropriate player or dealer if allowed by the rules. After
    a game is finished returns the payout for the game.

    Attributes:
        _currently_shown: A numpy array with shape (14, ) where each element
            corresponds to the number of cards discarded of the given card.
        _dealer: The dealer the players play against
        _shoe: The shoe to draw cards from.
        _focus: The identifier of the player currently in focus
        _payout: The payout for the game
        _players: A list of players that play the game.
        _resplit: The maximum number of players after resplitting
    """
    def __init__(self, shoe: Shoe):
        """ Initializes a game given a shoe.

        The game is initalized by drawing two shown cards for the player and a
        shown and a hidden card for the dealer.

        Args:
            shoe: The shoe to draw cards from.
        """
        self._shoe = shoe
        self._currently_shown = np.zeros(14, dtype=int)
        if self._shoe.try_reshuffle():
            self._currently_shown[0] = -1
        player_first = shoe.draw()
        player_second = shoe.draw()
        dealer_shown = shoe.draw()
        self._show_cards(player_first, player_second, dealer_shown)
        self._players = [Player(player_first,
                                player_second)]
        self._dealer = Dealer(dealer_shown,
                              shoe.draw())
        self._focus = 0
        if rule_variation.DEALER_PEEKS and self._dealer.get_blackjack():
            self._players[self._focus].stand()
        self._resplit = rule_variation.RESPLITTING_ACES if player_first == 1\
            else rule_variation.RESPLITTING_UPTO
        self._payout = None
        self._try_finish()

    def _all_player_stand(self) -> bool:
        """ Returns whether all players stand already.
        """
        for player in self._players:
            if not player.get_stand():
                return False
        return True

    def can_split(self) -> bool:
        """ Returns whether the player in focus can split.
        """
        if self._resplit <= len(self._players):
            return False
        if self._get_player() is None:
            return False
        card = self._get_player().split_value()
        if card is None or card not in rule_variation.PAIR_SPLITTING:
            return False
        return True

    def current_observation(self) -> Dict[str, np.ndarray]:
        """Returns an observation of the game.

        Returns:
            An observation of the current player's hand total, the number of
            aces which have the possibility to be 1 or 11, the dealer's hand
            total and the cards that have been revealed since the last action.
        """
        player = self._get_player()
        dealer = self._dealer
        if player is None:
            player = self._players[-1]
        currently_shown = self._currently_shown.copy()
        self._currently_shown = np.zeros(14, dtype=int)
        return {'PLAYER_TOTAL': np.array(player.get_total(), dtype=np.int),
                'PLAYER_ACES': np.array(player.get_aces(), dtype=np.int),
                'DEALER_TOTAL': np.array(dealer.get_total(), dtype=np.int),
                'REVEALED_CARDS': currently_shown
                }

    def double_focus(self) -> None:
        """Doubles down the hand of the player currently in focus if allowed.

        Return:
            A boolean value whether the player was allowed to double down.
        """
        if self._get_player() is None:
            return
        self._show_cards(self._get_player().double_down(self._shoe))
        self.move_focus()

    def get_payout(self) -> Optional[float]:
        return self._payout

    def _get_player(self) -> Optional[Player]:
        """Gets the player in focus

        Returns:
            The player in focus or None if no player is in focus.
        """
        if self._focus >= len(self._players) or self._focus < 0:
            return None
        return self._players[self._focus]

    def hit(self) -> None:
        """Hits the hand of the player currently in focus.
        """
        if self._get_player() is not None:
            self._show_cards(self._get_player().hit(self._shoe))
            if self._get_player().get_stand():
                self.move_focus()
        self._try_finish()

    def move_focus(self, to: int = None) -> None:
        """ Moves the focus.

        The focus is moved to the next player if no attribute is present,
        otherwise move the focus to the given index.

        Args:
            to: The index to move the focus to.
        """
        if to is None:
            to = self._focus + 1
        if to <= len(self._players):
            self._focus = to

    def player_in_focus(self) -> bool:
        """ Returns whether there is a player in focus. """
        return self._get_player() is not None

    def _show_cards(self, *cards: Card) -> None:
        """ Adds cards to the currently shown cards attribute.

        args:
            cards: The cards to add to the currently shown cards.
        """
        for card in cards:
            if card is not None:
                self._currently_shown[card] += 1

    def split_all(self, max_split: int) -> int:
        """Splits every player.

        Iterates through each player and splits their hand if the player is
        allowed to split and the number of players does not exceed the maximum
        number.

        Args:
            max_split: The maximum number of players allowed after splitting

        Returns:
            The number of players after splitting.
        """
        self.move_focus(0)
        while self._get_player() is not None:
            while self.can_split() and max_split > 1:
                max_split -= 1
                self.split_focus()
            self.move_focus()
        self.move_focus(0)
        return len(self._players)

    def split_focus(self) -> None:
        """ Splits the focus if allowed.
        """
        if not self.can_split():
            return
        focus = self._focus
        card = self._players[focus].split_value()
        can_double = rule_variation.DOUBLE_AFTER_SPLIT
        blackjack_with_aces = rule_variation.BLACKJACK_WITH_SPLIT_ACES
        player_new_first = self._shoe.draw()
        self._players[focus] = Player(card,
                                      player_new_first,
                                      can_double,
                                      card != 1 or blackjack_with_aces)
        player_new_second = self._shoe.draw()
        self._players += [Player(card,
                                 player_new_second,
                                 can_double,
                                 card != 1 or blackjack_with_aces)]
        self._show_cards(player_new_first, player_new_second)
        if card == 1 and not rule_variation.HIT_AFTER_SPLIT_ACES:
            self._players[focus].stand()
            self._players[-1].stand()
        self._try_finish()

    def stand(self) -> None:
        """Stands with the hand of the player currently in focus.

        Return:
            The payout if this move finished the game and None otherwise.
        """
        player = self._get_player()
        if player is not None:
            player.stand()
            self.move_focus()
        self._try_finish()

    def _try_finish(self) -> None:
        """Tries finishing the game.

        If every player is standing, deals the appropriate cards to the dealer
        and determines the payout for the current game.

        Returns:
            The payout for the game, if the game is not finished returns None.
        """
        if (self._payout is not None) or (not self._all_player_stand()):
            return
        player = self._get_player()
        dealer = self._dealer
        self._payout = 0.
        self._currently_shown += dealer.stand(self._shoe)
        for player in self._players:
            if player.get_blackjack():
                payout = (0.0 if dealer.get_blackjack() else
                          rule_variation.BLACKJACK_PAYOUT)
            else:
                payout = _get_payout(player.get_total(),
                                     dealer.get_total())
                if player.get_doubled_down():
                    payout *= 2
            self._payout += payout
