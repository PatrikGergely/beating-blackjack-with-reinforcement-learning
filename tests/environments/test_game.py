import numpy as np

from bbwrl.environments.game import Game
from bbwrl.environments.shoe import _custom_shoe


def test_can_split():
    shoe = _custom_shoe([10, 12, 5, 10])
    assert Game(shoe).can_split()
    shoe = _custom_shoe([5, 8, 5, 10])
    assert not Game(shoe).can_split()

    shoe = _custom_shoe([5, 5, 5, 10, 5, 5, 5, 5])
    game = Game(shoe)
    assert game.can_split()
    game.split_focus()
    assert game.can_split()
    game.split_focus()
    assert game.can_split()
    game.split_focus()
    assert not game.can_split()

    shoe = _custom_shoe([1, 1, 5, 10, 1, 1])
    game = Game(shoe)
    assert game.can_split()
    game.split_focus()
    assert not game.can_split()


def test_current_observation():
    shoe = _custom_shoe([10, 12, 5, 10])
    obs = Game(shoe).current_observation()
    assert obs['PLAYER_TOTAL'] == 20
    assert obs['PLAYER_ACES'] == 0
    assert obs['DEALER_TOTAL'] == 5
    assert np.array_equal(obs['REVEALED_CARDS'],
                          [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
