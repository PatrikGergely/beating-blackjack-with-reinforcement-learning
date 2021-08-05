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
""" Generates a basic strategist lookup table.

Generates a basic strategist lookup table, by calling an optimal strategist
using a full remaining shoe on every possible dealer and player hand
combination to determine the optimal move when no additional knowledge is
given to the player.
"""
from bbwrl.bot.strategists import OptimalStrategist
from bbwrl.environments import rule_variation

import numpy as np


def main():
    os = OptimalStrategist(lambda x: x)
    card_distribution = np.full((14,), 4*rule_variation.SHOE_SIZE)
    card_distribution[0] = 0
    # Ace strategy:
    print('BASIC_ACE_STRATEGY = [')
    print('    # 2    3    4    5    6    7    8    9    10   A')
    for player_total in range(12, 22):
        moves = []
        for dealer_card in range(2, 12):
            player_aces = 1
            if os.should_double(player_total,
                                player_aces,
                                dealer_card,
                                card_distribution):
                moves.append('D')
            elif os.should_hit(player_total,
                               player_aces,
                               dealer_card,
                               card_distribution):
                moves.append('H')
            else:
                moves.append('S')
        print('    {}, # {}'.format(moves, player_total))
    print(']\n')

    # Hit strategy:
    print('BASIC_HIT_STRATEGY = [')
    print('    # 2    3    4    5    6    7    8    9    10   A')
    for player_total in range(3, 22):
        moves = []
        for dealer_card in range(2, 12):
            player_aces = 0
            if os.should_double(player_total,
                                player_aces,
                                dealer_card,
                                card_distribution):
                moves.append('D')
            elif os.should_hit(player_total,
                               player_aces,
                               dealer_card,
                               card_distribution):
                moves.append('H')
            else:
                moves.append('S')
        print('    {}, # {}'.format(moves, player_total))
    print(']\n')

    # Split strategy:
    print('BASIC_SPLIT_STRATEGY = [')
    print('    #  2      3      4      5      6      7      8      9'
          '      10     A')
    # Player has aces
    moves = []
    for dealer_card in range(2, 12):
        player_total = 12
        player_aces = 1
        if os.should_split(player_total,
                           player_aces,
                           dealer_card,
                           card_distribution):
            moves.append('True ')
        else:
            moves.append('False')
    print('    [', end = '')
    print(*moves, sep = ', ', end = '] # As \n')

    for player_card in range(2, 11):
        moves = []
        for dealer_card in range(2, 12):
            player_total = 2*player_card
            player_aces = 0
            if os.should_split(player_total,
                               player_aces,
                               dealer_card,
                               card_distribution):
                moves.append('True ')
            else:
                moves.append('False')
        print('    [', end = '')
        print(*moves, sep = ', ', end = '], # {} \n'.format(player_total))
    print(']\n')



if __name__ == '__main__':
    main()
