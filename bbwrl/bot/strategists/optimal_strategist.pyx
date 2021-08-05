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
""" Implementation of the optimal strategist. """


# distutils: language=c++
from bbwrl.bot.reward_distribution cimport RewardDistribution
from bbwrl.bot.reward_distribution import RewardDistribution
from libcpp cimport bool
from bbwrl.environments import rule_variation


cdef class OptimalStrategist:
    """ Implementation of the optimal strategist.

    The strategist decides on the optimal action by comparing the reward
    distribution corresponding to the available actions.

    Attributes:
        reward_distribution: An object capable of computing distributions
            corresponding to different actions.
    """
    cdef RewardDistribution reward_distribution

    def __init__(self, utility_function):
        self.reward_distribution = RewardDistribution(utility_function)

    cpdef bool should_split(self,
                            int player_total,
                            int player_aces,
                            int dealer_total,
                            card_distribution):
        """ Returns whether it is optimal to split in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        cdef RewardDistribution rd = self.reward_distribution
        rd.set_card_distribution(card_distribution)
        cdef double distr_split[17]
        distr_split = rd.distr_split(player_total,
                                     player_aces,
                                     dealer_total)
        cdef double distr_hit_stand_double[17]
        distr_hit_stand_double = rd.distr_hit_stand_double(player_total,
                                                           player_aces,
                                                           dealer_total)
        return (rd.distribution_value(distr_split) >
                rd.distribution_value(distr_hit_stand_double))

    cpdef bool should_double(self,
                             int player_total,
                             int player_aces,
                             int dealer_total,
                             card_distribution):
        """ Returns whether it is optimal to double down in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        cdef RewardDistribution rd = self.reward_distribution
        rd.set_card_distribution(card_distribution)
        cdef double distr_double[17]
        distr_double = rd.distr_double(player_total,
                                       player_aces,
                                       dealer_total)
        cdef double distr_hit_stand[17]
        distr_hit_stand = rd.distr_hit_stand(player_total,
                                             player_aces,
                                             dealer_total)
        return (rd.distribution_value(distr_double) >
                rd.distribution_value(distr_hit_stand))

    cpdef bool should_hit(self,
                          int player_total,
                          int player_aces,
                          int dealer_total,
                          card_distribution):
        """ Returns whether it is optimal to hit in the given state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces available to the player.
            dealer_total: The hand total of the dealer.
            card_distribution: The distribution of the cards remaining in the
                shoe.
        """
        cdef RewardDistribution rd = self.reward_distribution
        rd.set_card_distribution(card_distribution)
        cdef double distr_hit[17]
        distr_hit = rd.distr_hit(player_total,
                                 player_aces,
                                 dealer_total)
        cdef double distr_stand[17]
        distr_stand = rd.distr_stand(player_total,
                                     1 if dealer_total == 11 else 0,
                                     dealer_total,
                                     True)
        return (rd.distribution_value(distr_hit) >
                rd.distribution_value(distr_stand))

    cpdef void free_mem(self):
        """ Frees the memory used by the Reward Distribution object. """
        self.reward_distribution.free_mem()
