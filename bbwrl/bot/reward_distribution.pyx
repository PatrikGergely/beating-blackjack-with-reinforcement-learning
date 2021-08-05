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
""" Implements the reward distribution class. """

# distutils: language=c++
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp cimport bool
cimport cython
from bbwrl.environments import rule_variation


cdef int get_value(int card):
    """ Return the value corresponding to a card. """
    if card == 1:
        return 11
    if card < 10:
        return card
    return 10

cdef int is_ace(int card_value):
    return 1 if card_value == 11 else 0

cdef void add(double* distr1, double* distr2, double scalar):
    """ Add two distributions together.

    Args:
        distr1: The distribution to add the other one to.
        distr2: The distribution to add to the other one.
        scalar: The scalar to multiply distr2 with.
    """
    cdef int i
    for i in range(17):
        distr1[i] += scalar*distr2[i]

cdef double* empty_distr():
    """ Creates an empty distribution. """
    cdef int i
    cdef double* distr = <double*> PyMem_Malloc(17 * sizeof(double))
    for i in range(17):
        distr[i] = 0.0
    return distr

cdef double* constant_distr(double value):
    """ Create constant distribution.

    Args:
        value: The value with probability 1 in the distribution.
    """
    cdef int i
    cdef double* distr = empty_distr()
    distr[int(value*2+8)] = 1.0
    return distr

cdef void double_distr(double* distr):
    """ Returns 2*distr. """
    cdef double tmp[17]
    cdef int i
    for i in range(17):
        tmp[i] = 0.0
    for i in range(4,13):
        tmp[2*i-8] = distr[i]
    for i in range(17):
        distr[i] = tmp[i]

cdef void double_variable(double* distr):
    """ Returns distr + distr. """
    cdef double tmp[17]
    cdef int i
    cdef int j
    for i in range(17):
        tmp[i] = 0.0
    for i in range(17):
        for j in range(17):
            if i + j < 25 and i+j >= 8:
                tmp[i + j - 8] += distr[i] * distr[j]
    for i in range(17):
        distr[i] = tmp[i]

cdef class RewardDistribution:
    """ Implements a class that computes the distribution of possible payouts.

    Determines the distribution of possible payouts based on the distribution
    of cards remaining in the shoe and hand of the dealer and total to any
    action taken by the player, by recursively calculating the optimal steps
    the player should take.
    """

    def __cinit__(self, utility_function):
        cdef int i
        for i in range(17):
            self.utility[i] = utility_function((i-8.0)/2.0)
        if rule_variation.BLACKJACK_PAYOUT != 1.5:
            print('blackjakc_payout_exception')
            # Raise error
            pass
        if rule_variation.SHOE_SIZE >= 25:
            print('shoe_size_exception')
            # Raise error
            pass
        if not rule_variation.DEALER_PEEKS:
            print('dealer_peeks_exception')
            # Raise error
            pass

        self.CONSTANT_TIE = constant_distr(0.0)
        self.CONSTANT_WIN = constant_distr(1.0)
        self.CONSTANT_BLACKJACK = constant_distr(1.5)
        self.CONSTANT_LOSE = constant_distr(-1.0)

    cdef void set_card_distribution(self, card_distribution):
        self.card_distr = card_distribution

    cdef double* distr_hit(self,
                           int player_total,
                           int player_aces,
                           int dealer_total):
        """ Returns the distribution corresponding to hitting in a state.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces of the player.
            dealer_total: The hand total of the dealer.
        """
        if player_total > 21:
            if player_aces > 0:
                # Soft hand becomes hard
                return self.distr_hit(player_total - 10,
                                      player_aces - 1,
                                      dealer_total)
            # Player busted
            return self.CONSTANT_LOSE
        cdef int128 state_hash = self.get_state_hash(player_total,
                                                     player_aces,
                                                     dealer_total,
                                                     2)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        for card in range(1,14):
            p = self.card_probability(card)
            if p > 0:
                self.card_distr[card] -= 1
                add(distr,
                    self.distr_hit_stand(player_total+get_value(card),
                                         player_aces+is_ace(get_value(card)),
                                         dealer_total),
                    p)
                self.card_distr[card] += 1
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_stand(self,
                             int player_total,
                             int dealer_aces,
                             int dealer_total,
                             bool first_call):
        """ Returns the distribution corresponding to standing in a state.

        Args:
            player_total: The hand total of the player.
            dealer_aces: The number of soft aces of the dealer.
            dealer_total: The hand total of the dealer.
            first_call: Whether the next card is the hidden card. If so due to
                the peek rule, the hidden card has some limitations.
        """
        if player_total > 21:
            # Player busted
            return self.CONSTANT_LOSE
        if dealer_total > 21:
            if dealer_aces > 0:
                # Soft hand becomes hard
                return self.distr_stand(player_total,
                                        dealer_aces - 1,
                                        dealer_total - 10,
                                        False)
            # Dealer busted
            return self.CONSTANT_WIN
        if dealer_total > 17 or (dealer_total == 17 and
                                 (dealer_aces == 0 or not rule_variation.HIT_SOFT_17)):
            #Stand
            if dealer_total == player_total:
                return self.CONSTANT_TIE
            if dealer_total > player_total:
                return self.CONSTANT_LOSE
            return self.CONSTANT_WIN
        cdef int128 state_hash = self.get_state_hash(player_total,
                                                     dealer_aces,
                                                     dealer_total,
                                                     3 if first_call else 4)
        if self.precomputed_distr(state_hash):
                return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        cdef int banned_value = 0
        if first_call:
            # Because of peek
            if dealer_total == 10:
                banned_value = 11
            if dealer_total == 11:
                banned_value = 10
        for card in range(1,14):
            p = self.card_probability(card, banned_value)
            if p > 0:
                self.card_distr[card] -= 1
                add(distr,
                    self.distr_stand(player_total,
                                     dealer_aces + is_ace(get_value(card)),
                                     dealer_total + get_value(card),
                                     False),
                    p)
                self.card_distr[card] += 1
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_double(self,
                              int player_total,
                              int player_aces,
                              int dealer_total):

        """ Returns the distribution corresponding to doubling down in a state.

        The distribution is calculated by calculating the distribution of
        hitting a single card then standing to gain distr, returning 2*distr.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces of the player.
            dealer_total: The hand total of the dealer.
        """
        cdef int128 state_hash
        state_hash = self.get_state_hash(player_total,
                                         player_aces,
                                         dealer_total,
                                         1)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        cdef int player_total_with_card
        for card in range(1,14):
            p = self.card_probability(card)
            if p > 0:
                self.card_distr[card] -= 1
                player_total_with_card = player_total + get_value(card)
                if (player_total_with_card > 21 and
                    player_aces + is_ace(get_value(card)) > 0):
                    player_total_with_card -= 10
                add(distr,
                    self.distr_stand(player_total_with_card,
                                     is_ace(dealer_total),
                                     dealer_total,
                                     True),
                     p)
                self.card_distr[card] += 1
        double_distr(distr)
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_split_general(self,
                             int player_card,
                             int dealer_total):
        """ Returns distribution corresponding to splitting a general card.

        The distribution is calculated by calculating the distribution using
        only one of the cards as a hand total to gain distr and then returning
        distr * distr.

        A general card is everything that isn't a ten, ace or face card.

        Args:
            player_card: The card the player wants to split.
            dealer_total: The hand total of the dealer.
        """
        state_hash = self.get_state_hash(player_card,
                                         0,
                                         dealer_total,
                                         0)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        for card in range(1,14):
            p = self.card_probability(card)
            if p > 0:
                self.card_distr[card] -= 1
                if rule_variation.DOUBLE_AFTER_SPLIT:
                    add(distr,
                        self.distr_hit_stand_double(player_card+get_value(card),
                                                    is_ace(get_value(card)),
                                                    dealer_total),
                        p)
                else:
                    add(distr,
                        self.distr_hit_stand(player_card+get_value(card),
                                             is_ace(get_value(card)),
                                             dealer_total),
                        p)
                self.card_distr[card] += 1
        double_variable(distr)
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_split_tens(self,
                                 int dealer_total):
        """ Returns distribution corresponding to splitting a ten or face card.

        The distribution is calculated by calculating the distribution using
        only one of the cards as a hand total to gain distr and then returning
        distr * distr.

        When the newly hit card is an ace, the player receives a blackjack.

        Args:
            dealer_total: The hand total of the dealer.
        """
        state_hash = self.get_state_hash(10,
                                         0,
                                         dealer_total,
                                         0)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        for card in range(2,14):
            p = self.card_probability(card)
            if p > 0:
                self.card_distr[card] -= 1
                if rule_variation.DOUBLE_AFTER_SPLIT:
                    add(distr,
                        self.distr_hit_stand_double(10 + get_value(card),
                                                    0,
                                                    dealer_total),
                        p)
                else:
                    add(distr,
                        self.distr_hit_stand(10 + get_value(card),
                                             0,
                                             dealer_total),
                        p)
                self.card_distr[card] += 1
        p = self.card_probability(1)
        if p > 0:
            add(distr, self.CONSTANT_BLACKJACK, p)
        double_variable(distr)
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_split_aces(self,
                                  int dealer_total):
        """ Returns distribution corresponding to splitting an ace.

        The distribution is calculated by calculating the distribution using
        only one of the cards as a hand total to gain distr and then returning
        distr * distr.

        When the newly hit card is a ten or face card, the player receives
        blackjack depending on the rule variation.

        Args:
            dealer_total: The hand total of the dealer.
        """

        state_hash = self.get_state_hash(11,
                                         1,
                                         dealer_total,
                                         0)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef int card
        cdef double p
        for card in range(1,10):
            p = self.card_probability(card)
            if p > 0:
                self.card_distr[card] -= 1
                if rule_variation.HIT_AFTER_SPLIT_ACES:
                    if rule_variation.DOUBLE_AFTER_SPLIT:
                        add(distr,
                            self.distr_hit_stand_double(11 + card,
                                                        1,
                                                        dealer_total),
                            p)
                    else:
                        add(distr,
                            self.distr_hit_stand(11 + card,
                                                 1,
                                                 dealer_total),
                            p)
                else:
                    add(distr,
                        self.distr_stand(11 + card,
                                         is_ace(dealer_total),
                                         dealer_total,
                                         True),
                        p)
                self.card_distr[card] += 1
        for card in range(10,14):
            p = self.card_probability(card)
            if p > 0:
                if rule_variation.BLACKJACK_WITH_SPLIT_ACES:
                    add(distr, self.CONSTANT_BLACKJACK, p)
                else:
                    self.card_distr[card] -= 1
                    add(distr,
                        self.distr_stand(21,
                                         is_ace(dealer_total),
                                         dealer_total,
                                         True),
                        p)
                    self.card_distr[card] += 1
        double_variable(distr)
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double* distr_split(self,
                             player_total,
                             player_aces,
                             dealer_total):
         """ Returns the distribution corresponding to splitting in a state.

         Args:
             player_total: The hand total of the player.
             player_aces: The number of soft aces of the player.
             dealer_total: The hand total of the dealer.
         """

         if player_aces > 0:
            return self.distr_split_aces(dealer_total)
         if player_total == 20:
            return self.distr_split_tens(dealer_total)
         return self.distr_split_general(player_total / 2, dealer_total)

    cdef double* distr_hit_stand(self,
                                 int player_total,
                                 int player_aces,
                                 int dealer_total):
        """ Returns the distribution when the player has the choice to hit.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces of the player.
            dealer_total: The hand total of the dealer.
        """
        if player_total > 21 and player_aces > 0:
            player_total -= 10
            player_aces -= 1
        cdef double* distr_hit = self.distr_hit(player_total,
                                                player_aces,
                                                dealer_total)
        cdef double* distr_stand = self.distr_stand(player_total,
                                                    is_ace(dealer_total),
                                                    dealer_total,
                                                    True)
        return distr_hit if (self.distribution_value(distr_hit) >
                             self.distribution_value(distr_stand)) else distr_stand

    cdef double* distr_hit_stand_double(self,
                                        int player_total,
                                        int player_aces,
                                        int dealer_total):
        """ Returns the distribution when the player can double douwn.

        Args:
            player_total: The hand total of the player.
            player_aces: The number of soft aces of the player.
            dealer_total: The hand total of the dealer.
        """
        if player_total > 21 and player_aces > 0:
            player_total -= 10
            player_aces -= 1
        if player_total == 21:
            return self.distr_blackjack(dealer_total)
        cdef double* distr_hit_stand = self.distr_hit_stand(player_total,
                                                            player_aces,
                                                            dealer_total)
        cdef double* distr_double = self.distr_double(player_total,
                                                      player_aces,
                                                      dealer_total)

        return distr_hit_stand if (self.distribution_value(distr_hit_stand) >
                                   self.distribution_value(distr_double)) else distr_double

    cdef double* distr_blackjack(self, int dealer_total):
        """ Returns the distribution when the player has blackjack.

        Args:
            dealer_total: The hand total of the dealer.
        """
        if dealer_total < 10:
            return self.CONSTANT_BLACKJACK
        cdef int128 state_hash
        state_hash = self.get_state_hash(21,
                                         1,
                                         dealer_total,
                                         5)
        if self.precomputed_distr(state_hash):
            return self.precomp_distr[state_hash]
        cdef double* distr = empty_distr()
        cdef double p
        if dealer_total == 10:
            p = self.card_probability(1)
        else:
            p = (self.card_probability(10)+
                 self.card_probability(11)+
                 self.card_probability(12)+
                 self.card_probability(13))
        add(distr, self.CONSTANT_TIE, p)
        add(distr, self.CONSTANT_BLACKJACK, 1-p)
        self.precomp_distr[state_hash] = distr
        return distr

    cdef double distribution_value(self, double* distr):
        """ Returns the utility corresponding to a distribution. """
        cdef int i
        cdef double q = 0.0
        for i in range(17):
            q += distr[i] * self.utility[i]
        return q

    cdef int128 get_state_hash(self,
                               int player_total,
                               int aces,
                               int dealer_total,
                               int mode):
        """ Hashes a state.

        Hashes a state in order to avoid recomputing the same distributions
        over and over again. Having the same player and dealer hand could lead
        to different distributions based on what the current action is, and
        therefore a mode argument is passed to differentiate between them.

        The hash can be stored in an int128 and by design it is a bijection.

        Mode 0: Split
        Mode 1: Double
        Mode 2: Hit
        Mode 3: Stand_first
        Mode 4: Stand_rest
        Mode 5: Blackjack
        """
        cdef int128 k = 10
        cdef int128 hash = mode
        cdef int i
        for i in range(1,14):
            hash += k * self.card_distr[i]
            k *= 100
        hash += k * player_total
        k *= 100
        hash += k * dealer_total
        k *= 100
        hash += k * aces
        return hash

    cdef bool precomputed_distr(self,
                                int128 hash):
        """ Returns if a distribution has been precomputed before.

        Args:
            hash: The hash of the distribution
        """
        return self.precomp_distr.find(hash) != self.precomp_distr.end()

    @cython.cdivision(True)
    cdef double card_probability(self, int card, int banned_value = 0):
        """ Get the probability of drawing a particular card.

        Args:
            card: The card to obtain the corresponding probability for.
            banned_value: A card value which is not allowed to be drawed next.
        """
        cdef double valid_card_distr[14]
        cdef double cards_left = 0
        cdef int i
        for i in range(1,14):
            valid_card_distr[i] = ( 0.0 if banned_value == get_value(i)
                                        else self.card_distr[i] )
            cards_left += valid_card_distr[i]
        return valid_card_distr[card] / cards_left

    cdef void free_mem(self):
        """ Frees the allocated memory. """
        cdef pair[int128, double*] item
        for item in self.precomp_distr:
            PyMem_Free(item.second)
        self.precomp_distr.clear()
        self.precomp_distr.rehash(0)

    def __dealloc__(self):
        """ Deallocated the object. """
        self.free_mem()
        PyMem_Free(self.CONSTANT_TIE)
        PyMem_Free(self.CONSTANT_LOSE)
        PyMem_Free(self.CONSTANT_WIN)
        PyMem_Free(self.CONSTANT_BLACKJACK)
