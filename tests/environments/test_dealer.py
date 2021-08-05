import numpy as np

from bbwrl.environments.game import Dealer
from bbwrl.environments.shoe import _custom_shoe


def test_blackjack():
    assert Dealer(1, 12).get_blackjack()
    assert not Dealer(11, 10).get_blackjack()
    dealer = Dealer(1, 5)
    dealer._add_card(5)
    assert not dealer.get_blackjack()

def test_aces():
    assert Dealer(1, 8).get_aces() == 1
    assert Dealer(2, 8).get_aces() == 0
    dealer = Dealer(3, 3)
    assert dealer.get_aces() == 0
    dealer._add_card(1)
    assert dealer.get_aces() == 1
    dealer._add_card(1)
    assert dealer.get_aces() == 1
    dealer._add_card(11)
    assert dealer.get_aces() == 0


def test_total():
    assert Dealer(1, 8).get_total() == 11
    assert Dealer(2, 8).get_total() == 2
    dealer = Dealer(3, 3)
    assert dealer.get_total() == 3
    dealer._add_card(1)
    assert dealer.get_total() == 14
    dealer._add_card(1)
    assert dealer.get_total() == 15
    dealer._add_card(11)
    assert dealer.get_total() == 15


def test_stand():
    shoe = _custom_shoe([3, 10, 4, 11])
    dealer = Dealer(1, 6)
    assert np.array_equal(dealer.stand(shoe),
                          [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    assert dealer.get_total() == 17
    dealer = Dealer(1, 2)
    assert np.array_equal(dealer.stand(shoe),
                          [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    assert dealer.get_total() == 20
    dealer = Dealer(12, 4)
    assert np.array_equal(dealer.stand(shoe),
                          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    assert dealer.get_total() == 24
