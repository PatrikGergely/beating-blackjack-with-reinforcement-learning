from bbwrl.environments.game import Player
from bbwrl.environments.shoe import _custom_shoe


def test_double_down():
    shoe = _custom_shoe([5, 3])
    assert Player(10, 3, False).double_down(shoe) is None
    player = Player(10, 3, True)
    assert player.double_down(shoe) == 5
    assert player.get_total() == 18
    assert player.get_doubled_down()
    player = Player(3, 3)
    player.hit(shoe)
    assert player.double_down(shoe) is None


def test_blackjack():
    assert Player(1, 12).get_blackjack()
    assert not Player(1, 12, False, False).get_blackjack()
    assert not Player(11, 10).get_blackjack()
    player = Player(1, 5)
    player._add_card(5)
    assert not player.get_blackjack()


def test_aces():
    assert Player(1, 8).get_aces() == 1
    assert Player(2, 8).get_aces() == 0
    player = Player(3, 3)
    assert player.get_aces() == 0
    player._add_card(1)
    assert player.get_aces() == 1
    player._add_card(1)
    assert player.get_aces() == 1
    player._add_card(11)
    assert player.get_aces() == 0


def test_total():
    assert Player(1, 8).get_total() == 19
    assert Player(2, 8).get_total() == 10
    player = Player(3, 3)
    assert player.get_total() == 6
    player._add_card(1)
    assert player.get_total() == 17
    player._add_card(1)
    assert player.get_total() == 18
    player._add_card(11)
    assert player.get_total() == 18


def test_hit():
    shoe = _custom_shoe([5, 10, 4])
    player = Player(2, 3, True)
    assert player.hit(shoe) == 5
    assert not player.get_stand()
    assert player.hit(shoe) == 10
    assert not player.get_stand()
    assert player.hit(shoe) == 4
    assert player.get_stand()
    assert player.hit(shoe) is None


def test_stand():
    shoe = _custom_shoe([5, 10])
    player = Player(2, 3, True)
    assert player.hit(shoe) == 5
    player.stand()
    assert player.hit(shoe) is None
    assert player.get_stand()


def test_split_values():
    assert Player(10, 12).split_value() == 10
    assert Player(1, 11).split_value() is None
    player = Player(3, 3)
    shoe = _custom_shoe([3])
    player.hit(shoe)
    assert player.split_value() is None
