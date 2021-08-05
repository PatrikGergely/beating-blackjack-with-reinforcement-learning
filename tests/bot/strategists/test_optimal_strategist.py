import numpy as np

import bbwrl.utils.pyxinstall
from bbwrl.bot.strategists.optimal_strategist import OptimalStrategist


def _create_card_distr():
    return np.array([0, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16])


def test_should_split():
    strategist = OptimalStrategist(lambda x: x)
    assert not strategist.should_split(10, 0, 6, _create_card_distr())
    assert not strategist.should_split(20, 0, 6, _create_card_distr())
    assert strategist.should_split(12, 1, 6, _create_card_distr())
    assert strategist.should_split(16, 0, 6, _create_card_distr())


def test_should_double():
    strategist = OptimalStrategist(lambda x: x)
    assert not strategist.should_double(14, 0, 9, _create_card_distr())
    assert not strategist.should_double(20, 1, 6, _create_card_distr())
    assert strategist.should_double(18, 1, 6, _create_card_distr())


def test_should_hit():
    strategist = OptimalStrategist(lambda x: x)
    assert strategist.should_hit(14, 0, 9, _create_card_distr())
    assert not strategist.should_hit(20, 1, 6, _create_card_distr())
    assert strategist.should_hit(17, 1, 6, _create_card_distr())
