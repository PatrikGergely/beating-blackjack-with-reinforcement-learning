""" Wrapper class needed to test RewardDistribution. """

cimport bbwrl.bot.reward_distribution as rd
import bbwrl.bot.reward_distribution
import numpy as np
cimport numpy as np
from libcpp cimport bool
from cpython.mem cimport PyMem_Malloc, PyMem_Free

DISTR_SIZE = 17

cdef extern from *:
    ctypedef int int128 "__int128_t"

cdef np.ndarray[np.double_t, ndim=1] pointer_to_numpy(double* pointer_array, int size):
    return np.asarray(<np.double_t[:size]> pointer_array)

cdef double* numpy_to_pointer(np.ndarray[np.double_t, ndim=1] np_array):
    cdef double* pointer_array = <double*> PyMem_Malloc(np_array.size * sizeof(double))
    cdef int i
    for i in range(np_array.size):
        pointer_array[i] = np_array[i]
    return pointer_array

cpdef int get_value(int card):
    return rd.get_value(card)

cpdef int is_ace(int card):
    return rd.is_ace(card)

cpdef np.ndarray[np.double_t, ndim=1] add(np.ndarray[np.double_t, ndim=1] distr1, np.ndarray[np.double_t, ndim=1] distr2, double scalar):
    cdef double* pointer_distr1 = numpy_to_pointer(distr1)
    cdef double* pointer_distr2 = numpy_to_pointer(distr2)
    rd.add(pointer_distr1, pointer_distr2, scalar)
    cdef np.ndarray[np.double_t, ndim=1] ans = pointer_to_numpy(pointer_distr1, distr1.size)
    PyMem_Free(pointer_distr2)
    return ans

cpdef np.ndarray[np.double_t, ndim=1] empty_distr():
    return pointer_to_numpy(rd.empty_distr(), DISTR_SIZE)

cpdef np.ndarray[np.double_t, ndim=1] constant_distr(double value):
    return pointer_to_numpy(rd.constant_distr(value), DISTR_SIZE)

cpdef np.ndarray[np.double_t, ndim=1] double_distr(np.ndarray[np.double_t, ndim=1] distr):
    cdef double* pointer_distr = numpy_to_pointer(distr)
    rd.double_distr(pointer_distr)
    return pointer_to_numpy(pointer_distr, DISTR_SIZE)

cpdef np.ndarray[np.double_t, ndim=1] double_variable(np.ndarray[np.double_t, ndim=1] distr):
    cdef double* pointer_distr = numpy_to_pointer(distr)
    rd.double_variable(pointer_distr)
    return pointer_to_numpy(pointer_distr, DISTR_SIZE)

cdef class WrappedRewardDistribution:
    cdef rd.RewardDistribution reward_distr

    def __init__(self, utility_function):
        self.reward_distr = rd.RewardDistribution(utility_function)

    cpdef void set_card_distribution(self, card_distribution):
        self.reward_distr.set_card_distribution(card_distribution)

    cpdef np.ndarray[np.double_t, ndim=1] distr_hit(self,
                             int player_total,
                             int player_aces,
                             int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_hit(player_total,
                                                            player_aces,
                                                            dealer_total),
                                17)

    cpdef np.ndarray[np.double_t, ndim=1] distr_stand(self,
                             int player_total,
                             int dealer_aces,
                             int dealer_total,
                             bool first_call):
        return pointer_to_numpy(self.reward_distr.distr_stand(player_total,
                                                              dealer_aces,
                                                              dealer_total,
                                                              first_call),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_double(self,
                              int player_total,
                              int player_aces,
                              int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_double(player_total,
                                                               player_aces,
                                                               dealer_total),
                                DISTR_SIZE)


    cpdef np.ndarray[np.double_t, ndim=1] distr_split_general(self,
                             int player_card,
                             int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_split_general(player_card,
                                                                dealer_total),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_split_tens(self,
                                  int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_split_tens(dealer_total),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_split_aces(self,
                                  int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_split_aces(dealer_total),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_split(self,
                             player_total,
                             player_aces,
                             dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_split(player_total,
                                                              player_aces,
                                                              dealer_total),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_hit_stand(self,
                                 int player_total,
                                 int player_aces,
                                 int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_hit_stand(player_total,
                                                                  player_aces,
                                                                  dealer_total),
                                DISTR_SIZE)

    cpdef np.ndarray[np.double_t, ndim=1] distr_hit_stand_double(self,
                                        int player_total,
                                        int player_aces,
                                        int dealer_total):
        return pointer_to_numpy(self.reward_distr.distr_hit_stand_double(player_total,
                                                                         player_aces,
                                                                         dealer_total),
                                DISTR_SIZE)

    cpdef double distribution_value(self, np.ndarray[np.double_t, ndim=1] distr):
        return self.reward_distr.distribution_value(numpy_to_pointer(distr))

    cpdef int128 get_state_hash(self,
                               int player_total,
                               int aces,
                               int dealer_total,
                               int mode):
        return self.reward_distr.get_state_hash(player_total,
                                                aces,
                                                dealer_total,
                                                mode)

    cpdef bool precomputed_distr(self,
                                int128 hash):
        return self.reward_distr.precomputed_hash(hash)

    cpdef double card_probability(self, int card, int banned_value):
        return self.reward_distr.card_probability(card, banned_value)

    cpdef void free_mem(self):
        self.free_mem()

    cpdef double[:] ASD(self):
        cdef double[:] a = np.zeros(17)
        a[10] = 1.0
        return a
