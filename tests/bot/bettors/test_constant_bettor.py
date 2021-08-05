import numpy as np

from bbwrl.bot.bettors.constant_bettor import ConstantBettor


def _create_card_distr():
    return np.array([0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])


def test_get_bet_size():
    bettor = ConstantBettor()
    assert bettor.get_bet_size(500.0, _create_card_distr()) == 1.0
    assert bettor.get_bet_size(255.0, _create_card_distr()) == 1.0
