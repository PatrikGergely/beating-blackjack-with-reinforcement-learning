"""Vegas Strip rule variation implementation.

Blackjack has many different variations and the following constants determine
the precise rules for the most popular rule variation, Vegas Strip.

Read further about the rule variations on the following websites:
    - https://www.blackjackgala.com/vegas-strip-blackjack/
    - https://www.blackjackexpert.com/variations/vegas-strip-blackjack/
"""

# The starting bankroll of the player.
AGENT_CHIPS = 600.0
# The payout when a player is dealt blackjack.
BLACKJACK_PAYOUT = 1.5
# Whether receiving blackjack after splitting aces rewards extra payout.
BLACKJACK_WITH_SPLIT_ACES = False
# Whether the dealer peeks after being dealt an ace, ten or face card.
DEALER_PEEKS = True
# Whether the player is allowed to double down after splitting their hand.
DOUBLE_AFTER_SPLIT = True
# Whether the player is allowed to hit after splitting aces.
HIT_AFTER_SPLIT_ACES = False
# Whether the dealer hits soft 17.
HIT_SOFT_17 = False
# A list of cards that is allowed to be split.
PAIR_SPLITTING = list(range(1, 14))
# The ratio of cards left to the whole shoe that when meet reshuffles the shoe.
RESHUFFLE = 0.25
# The number of hands the player can have after splitting aces.
RESPLITTING_ACES = 2
# The number of hands the player can have after splitting non-aces.
RESPLITTING_UPTO = 4
# The number of 52-cardcard decks in the shoe.
SHOE_SIZE = 4
# Whether splitting different face cards are allowed (such as J+Q).
SPLIT_UNEVEN = True
