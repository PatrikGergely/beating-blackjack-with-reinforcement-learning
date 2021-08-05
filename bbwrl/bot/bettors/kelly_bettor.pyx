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
""" Implements the Kelly bettor. """

# distutils: language=c++
from bbwrl.bot.reward_distribution cimport RewardDistribution
from bbwrl.bot.reward_distribution import RewardDistribution
from libcpp cimport bool
cimport cython
import sympy
from bbwrl.environments import rule_variation


MINN = -1e60


cdef int get_value(int card):
    """ Return the value corresponding to a card. """
    if card == 1:
        return 11
    if card < 10:
        return card
    return 10

cdef int is_ace(int card):
    return 1 if card == 1 else 0

@cython.cdivision(True)
cdef double get_probability(long[:] distr,
                            int card1,
                            int card2,
                            int card3):
    """ Return the probability of drawing 3 particular cards.

    Given a distribution of cards remaining in the shoe, calculate the
    probability of drawing 3 cards in order.

    Args:
        distr: The distribution of cards remaining in the shoe.
        card1: The first card to draw.
        card2: The second card to draw.
        card3: The third card to draw.
    """
    cdef double cards_left = 0
    cdef int i
    for i in range(1,14):
        cards_left += distr[i]
    if cards_left < 3:
        return 0
    cdef double p
    p = distr[card1] / cards_left
    distr[card1] -= 1
    p *= distr[card2] / (cards_left-1)
    distr[card2] -= 1
    p *= distr[card3] / (cards_left-2)
    distr[card1] += 1
    distr[card2] += 1
    return p

cdef (int, int) get_player(int player_first, int player_second):
    """ Get the hand total and number of soft aces corresponding to two cards.

    Args:
        player_first: The first initial card dealt to the player.
        player_second: The second initial card dealt to the player.

    Returns:
        A pair of integers, the first one corresponding to the hand total and
        the second one corresponding to the number of soft aces.
    """
    cdef int player_total = get_value(player_first) + get_value(player_second)
    cdef int player_aces = is_ace(player_first) + is_ace(player_second)
    if player_total > 21:
        player_total -= 10
        player_aces -= 1
    return player_total, player_aces

cdef long[:] remove_cards(long[:] distr,
                          int card1,
                          int card2,
                          int card3):
    """ Remove 3 cards from a card distribution.

    Args:
        distr: The distribution to remove the cards from.
        card1: The first card to remove from the distribution.
        card2: The second card to remove from the distribution.
        card3: The third card to remove from the distribution.

    Returns:
        A distribution with the 3 cards removed.
    """
    cdef long modified_card_distr[14]
    cdef int i
    for i in range(14):
        modified_card_distr[i] = distr[i]
    modified_card_distr[card1] -= 1
    modified_card_distr[card2] -= 1
    modified_card_distr[card3] -= 1
    return modified_card_distr

cdef void add(double* distr1, double* distr2, double scalar):
    """ Add two reward distributions together.

    Args:
        distr1: The distribution to add the other one to.
        distr2: The distribution to add to the other one.
        scalar: The scalar to multiply distr2 with.
    """
    cdef int i
    for i in range(17):
        distr1[i] += scalar*distr2[i]

cdef double find_maximum(sympy.core.expr.Expr f,
                         double cur_best,
                         double: x):
    """ Returns one of the two arguments with higher value in f.

    Finds the global maximum of a function, by repeadetly calling this function
    with the current maximum and the next point which has 0 as its derivative.

    Args:
        f: The function to maximize.
        cur_best: The current real number with the highest value.
        x: The real number to compare to the current maximum.

    Returns:
        x or cur_best depending on which takes a higher value in f.
    """
    best_y = sympy.limit(f, sympy.symbols('x'), cur_best)
    if not best_y.is_real:
        best_y = MINN
    y = sympy.limit(f, sympy.symbols('x'), x)
    if not y.is_real:
        y = MINN
    if y > best_y:
        return x
    return cur_best


cdef double get_dealer_blackjack_probability(long[:] card_distr,
                                             int dealer_total):
    """ Returns the probability that a dealer is dealt blackjack.

    Calculates the probability that the dealer has a hidden card that makes a
    blackjack with the shown card.

    Args:
        card_distr: The distribution of cards remaining in the shoe.
        dealer_total: The dealers shown card.
    """
    cdef double cards_left = 0
    cdef int i
    for i in range(1,14):
        cards_left += card_distr[i]
    if dealer_total == 10:
        return card_distr[1]/cards_left
    return (card_distr[10]+
            card_distr[11]+
            card_distr[12]+
            card_distr[13])/cards_left

cdef class KellyBettor:
    """ Implements a bettor maximizing long term gains.

    Impements a bettor that produces bet sizes that optimise for log(1+x),
    where x is the bankroll of the bot.

    Attributes:
        reward_distribution: An object capable of computing distributions
            corresponding to different actions.
    """
    cdef RewardDistribution reward_distribution

    def __init__(self):
        self.reward_distribution = RewardDistribution(lambda x: x)

    cpdef get_distr(self, long[:] card_distr):
        """ Returns the reward distribution when following the optimal strategy.

        Given a distribution of cards remaining in the deck calculates each
        possible dealt hand, and the reward distribution corresponding to it
        and adds these together after scaling with the probability of getting
        dealt a given hand.

        Args:
            card_distr: The distribution of cards remaining in the shoe.
        """
        cdef RewardDistribution rd = self.reward_distribution
        cdef double distr[17]
        cdef int i
        for i in range(17):
            distr[i] = 0.0
        cdef double current_distr[17]
        cdef double split_distr[17]
        cdef double p,q
        cdef long[:] modified_card_distr
        for player_first in range(1,14):
            for player_second in range(1, 14):
                for dealer_shown in range(1, 14):
                    p = get_probability(card_distr, player_first, player_second, dealer_shown)
                    if p == 0.0:
                        continue
                    player_total, player_aces = get_player(player_first, player_second)
                    dealer_total = get_value(dealer_shown)
                    modified_card_distr = remove_cards(card_distr, player_first, player_second, dealer_shown)
                    rd.set_card_distribution(modified_card_distr)
                    current_distr = rd.distr_hit_stand_double(player_total,
                                                              player_aces,
                                                              dealer_total)
                    q = 0.0 if dealer_total < 10 else get_dealer_blackjack_probability(card_distr,
                                                                                       dealer_total)
                    if matching_cards(player_first, player_second):
                        split_distr = rd.distr_split(player_total,
                                                     player_aces,
                                                     dealer_total)
                        if (rd.distribution_value(split_distr) >
                            rd.distribution_value(current_distr)):
                            current_distr = split_distr
                    add(distr, current_distr, p*(1-q))
                    add(distr, rd.CONSTANT_TIE if player_total == 21 else rd.CONSTANT_LOSE, p*q)
        rd.free_mem()
        return distr


    cpdef double get_bet_size(self,
                              double chips,
                              long[:] card_distriution):
        """ Returns the bet size that maximizes E[log(1+chips+payout*bet_size)]

        After calculating the reward distribution corresponding to the
        distribution of remaining cards in the deck and maximizes
        E[log(1+chips+payout*bet_size)] by differentiating the function with
        respect to bet_size and analyzing the stationary points.

        Args:
            chips: The bankroll of the player.
            card_distribution: The distribution of cards remaining in the shoe.
        """
        cdef double distr[17]
        distr = self.get_distr(card_distriution)
        cdef int i
        cdef double w
        cdef double p
        x = sympy.symbols('x')
        log_util = 0
        for i in range(17):
            w = (i-8.0)/2.0
            p = distr[i]
            log_util += p*sympy.log(1.0+chips+w*x)
        diff_log_util = log_util.diff(x)
        possible_maximums = sympy.solve(diff_log_util)
        ans = 1.0
        ans = find_maximum(log_util, ans, chips)
        for possible_maximum in possible_maximums:
            if possible_maximum.is_real and possible_maximum >= 1.0 and possible_maximum <= chips:
                ans = find_maximum(log_util, ans, possible_maximum)
        return ans


    cdef bool matching_cards(int card1, int card2):
        """ Returns whether two cards can be split or not."""
        if rule_variation.SPLIT_UNEVEN:
            return get_value(card1) == get_value(card2)
        return card1 == card2


    cpdef void set_payout(self, payout, card_distr):
        return

    cpdef void save(self):
        return
