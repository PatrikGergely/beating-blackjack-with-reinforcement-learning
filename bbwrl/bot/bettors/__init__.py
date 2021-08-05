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
""" Bettor implementations. """

from bbwrl.bot.bettors.constant_bettor import ConstantBettor
from bbwrl.utils import pyxinstall
from bbwrl.bot.bettors.kelly_bettor import KellyBettor  # type: ignore
from bbwrl.bot.bettors.vector_bettor import VectorBettor
from bbwrl.bot.bettors.ml.ddpg_bettor import DDPGBettor
from bbwrl.bot.bettors.ml.ddpg_bettor import DDPGTrainer
from bbwrl.bot.bettors.ml.dqn_bettor import DQNBettor
from bbwrl.bot.bettors.ml.dqn_bettor import DQNTrainer

# Dictionary mapping bettor names to the corresponding classes.
BETTORS = {
    'ConstantBettor': ConstantBettor,
    'KellyBettor': KellyBettor,
    'VectorBettor': VectorBettor,
    'DDPGBettor': DDPGBettor,
    'DDPGTrainer': DDPGTrainer,
    'DQNBettor': DQNBettor,
    'DQNTrainer': DQNTrainer,
}