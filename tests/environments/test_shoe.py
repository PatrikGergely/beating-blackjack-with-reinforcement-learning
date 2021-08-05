from bbwrl.environments.shoe import Shoe, _custom_shoe


def test_cards_left():
    shoe = Shoe()
    for _ in range(52):
        shoe.draw()
    assert shoe.cards_left() == 0.75
    for _ in range(52):
        shoe.draw()
    assert shoe.cards_left() == 0.5


def test_draw():
    shoe = _custom_shoe([1,2,3,4])
    assert shoe.draw() == 1
    assert shoe.draw() == 2
    assert shoe.draw() == 3
    assert shoe.draw() == 4

def test_reshuffle():
    shoe = Shoe()
    for _ in range(52):
        shoe.draw()
    assert shoe.cards_left() == 0.75
    shoe.reshuffle()
    assert shoe.cards_left() == 1.0


def test_try_reshuffle():
    shoe = Shoe()
    for _ in range(52):
        shoe.draw()
    assert not shoe.try_reshuffle()
    for _ in range(52):
        shoe.draw()
    assert not shoe.try_reshuffle()
    for _ in range(52):
        shoe.draw()
    assert not shoe.try_reshuffle()
    shoe.draw()
    assert shoe.try_reshuffle()
    shoe.reshuffle()
    assert not shoe.try_reshuffle()
