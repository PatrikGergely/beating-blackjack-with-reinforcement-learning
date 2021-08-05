import numpy as np

from bbwrl.bot.bettors.kelly_bettor import KellyBettor


def _create_card_distr():
    return np.array([0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])


def test_get_bet_size():
    bettor = KellyBettor()
    assert bettor.get_bet_size(
        500.0,
        np.array([0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
    ) == 1.0
    assert bettor.get_bet_size(
        500.0,
        np.array([0, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]),
    )- 52.36 <= 0.01
