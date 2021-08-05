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


# distutils: language=c++
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp cimport bool
cimport cython


cdef extern from *:
    ctypedef int int128 "__int128_t"

cdef int get_value(int card)

cdef int is_ace(int card)

cdef void add(double* distr1, double* distr2, double scalar)

cdef double* empty_distr()

cdef double* constant_distr(double value)

cdef void double_distr(double* distr)

cdef void double_variable(double* distr)

cdef class RewardDistribution:
    cdef double utility[17]
    cdef int card_distr[14]
    cdef unordered_map[int128, double*] precomp_distr
    cdef double* CONSTANT_TIE
    cdef double* CONSTANT_LOSE
    cdef double* CONSTANT_WIN
    cdef double* CONSTANT_BLACKJACK

    cdef void set_card_distribution(self, card_distribution)

    cdef double* distr_hit(self,
                           int player_total,
                           int player_aces,
                           int dealer_total)

    cdef double* distr_stand(self,
                             int player_total,
                             int dealer_aces,
                             int dealer_total,
                             bool first_call)

    cdef double* distr_double(self,
                              int player_total,
                              int player_aces,
                              int dealer_total)

    cdef double* distr_split_general(self,
                             int player_card,
                             int dealer_total)

    cdef double* distr_split_tens(self,
                                  int dealer_total)

    cdef double* distr_split_aces(self,
                                  int dealer_total)

    cdef double* distr_split(self,
                             player_total,
                             player_aces,
                             dealer_total)

    cdef double* distr_hit_stand(self,
                                 int player_total,
                                 int player_aces,
                                 int dealer_total)

    cdef double* distr_hit_stand_double(self,
                                        int player_total,
                                        int player_aces,
                                        int dealer_total)

    cdef double* distr_blackjack(self, int dealer_total)

    cdef double distribution_value(self, double* distr)

    cdef int128 get_state_hash(self,
                               int player_total,
                               int aces,
                               int dealer_total,
                               int mode)

    cdef bool precomputed_distr(self,
                                int128 hash)

    cdef double card_probability(self, int card, int banned_value = *)

    cdef void free_mem(self)
