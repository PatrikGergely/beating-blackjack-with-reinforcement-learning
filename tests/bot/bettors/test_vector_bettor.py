import numpy as np

from bbwrl.bot.bettors.vector_bettor import VectorBettor


def _create_card_distr():
    return


def test_get_bet_size():
    bettor = VectorBettor(4, [0, -1, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1])
    assert bettor.get_bet_size(
        500.0,
        np.array([0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]),
    ) == -1.0
    assert bettor.get_bet_size(
        500.0,
        np.array([0, 16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]),
    ) == 31.5
