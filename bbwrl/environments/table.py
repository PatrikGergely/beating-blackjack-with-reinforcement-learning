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

"""Blackjack table implementation."""
from typing import Any, Dict, Optional

import dm_env
from dm_env import specs
import numpy as np

from bbwrl.environments import rule_variation
from bbwrl.environments.game import Game
from bbwrl.environments.shoe import Shoe

_DEFAULT_GAME_OBS = {'PLAYER_TOTAL': np.array(0, dtype=np.int),
                     'PLAYER_ACES': np.array(0, dtype=np.int),
                     'DEALER_TOTAL': np.array(0, dtype=np.int),
                     'REVEALED_CARDS': np.zeros(14, dtype=np.int)
                     }


class Table(dm_env.Environment):
    """Represents a blackjack table in a casino.

    Implements the dm_env.Environment interface in order to allow agents to
    interact with the blackjack environment.

    Attributes:
        _chips: The amount of chips in the player's bankroll.
        _game: The game that is currently being played.
        _game_counter: The number of games played
        _shoe: The shoe where the cards are drawn from.
        _stage: The current stage of blackjack.
        _time_limit: The number of games to automatically terminate after.
        _stage2move: Maps stages to the corresponding method to call.
    """
    def __init__(self, time_limit: int):
        """Initializes a table.

        Args:
            _time_limit: The number of games to automatically terminate after.
        """
        self._stage2move = {
            'CHOOSE_BET': self._place_bet,
            'SPLIT?': self._split,
            'DOUBLE?': self._double,
            'HIT/STAND': self._hit_or_stand,
        }
        self._time_limit = time_limit
        self.reset()

    def reset(self) -> dm_env.TimeStep:
        """Returns the first `TimeStep` of a new episode."""
        self._shoe = Shoe()
        self._chips = rule_variation.AGENT_CHIPS
        self._stage = 'CHOOSE_BET'
        self._game: Optional[Game] = None
        self._game_counter = 1
        return dm_env.restart(self._observation())

    def step(self, action: float) -> dm_env.TimeStep:
        """Updates the environment according to the action.

        Args:
            action: The action taken by the agent.

        Returns:
            The next `TimeStep`.
        """
        self._stage2move[self._stage](action)
        return self._proceed()

    def action_spec(self) -> specs.Array:
        """Returns the action spec."""
        return specs.BoundedArray((), np.float32, minimum=0.0, maximum=np.inf)

    def observation_spec(self) -> Dict[str, specs.Array]:
        """Returns the observation spec."""
        return {'STAGE':
                specs.StringArray(()),
                'CHIPS':
                specs.BoundedArray(
                    (), np.float32, minimum=0.0, maximum=np.inf),
                'REVEALED_CARDS':
                specs.BoundedArray(
                    (14,), np.int, minimum=0,
                    maximum=4*rule_variation.SHOE_SIZE),
                'PLAYER_TOTAL':
                    specs.DiscreteArray(dtype=int, num_values=31),
                'PLAYER_ACES':
                    specs.DiscreteArray(dtype=int, num_values=2),
                'DEALER_TOTAL':
                    specs.DiscreteArray(dtype=int, num_values=11)
                }

    def _observation(self) -> Dict[str, Any]:
        """Returns an observation of the table.

        Returns:
            An observation of the current game, stage and amount of chips.
        """
        obs = _DEFAULT_GAME_OBS if self._game is None else \
            self._game.current_observation()
        obs['STAGE'] = self._stage
        obs['CHIPS'] = self._chips
        return obs

    def _proceed(self) -> dm_env.TimeStep:
        """ Proceeds to the next time step.

        Checks whether the game terminated. If not it transitions to the next
        timestep without any return, otherwise it returns the payout as reward
        and resets the game and updates the chips of the agent.
        """
        payout = self._game.get_payout()
        if payout is not None:
            self._chips += payout * self._bet_size
            if self._should_terminate():
                observation = self._observation()
                return dm_env.termination(reward=payout * self._bet_size,
                                          observation=observation)
            self._game_counter += 1
            self._stage = 'CHOOSE_BET'
            observation = self._observation()
            self._game = None
            return dm_env.transition(
                reward=payout * self._bet_size,
                observation=observation)
        return dm_env.transition(reward=0., observation=self._observation())

    def _should_terminate(self) -> bool:
        """ Returns whether the episode should be terminated.

        The episode should terminate if the agent has less chips than what is
        needed to bet or a sufficient number of games have been played already.
        """
        return self._chips < 1.0 or self._game_counter >= self._time_limit

    def _place_bet(self, bet_size: float) -> None:
        """ Places a bet for the next game.

        Args:
            bet_size: The bet the agent wants to place. This needs to be
                atleast 1.0 and can be atmost the chips available to the agent.
        """
        self._bet_size = max(min(bet_size, self._chips), 1.0)
        self._bet_multiplier = 1
        self._game = Game(self._shoe)
        if self._can_split():
            self._stage = 'SPLIT?'
        elif self._can_bet_more():
            self._stage = 'DOUBLE?'
        else:
            self._stage = 'HIT/STAND'

    def _split(self, want_to_split: float) -> None:
        """Splits the hand of the player if the agent asks to.

        If the action provided asks for the hand to be split, it splits as
        many times as possible.

        Args:
            want_to_split: A float representing whether the agent wants to
                split with the player in focus.
        """
        if bool(want_to_split):
            self._bet_multiplier = self._game.split_all(self._max_multiplier())
        if self._can_bet_more():
            self._stage = 'DOUBLE?'
        else:
            self._stage = 'HIT/STAND'

    def _double(self, want_to_double: float) -> None:
        """Double down the hand of the player if the agent asks to.

        Args:
            want_to_double: A float representing whether the agent wants to
                double down with the player in focus.
        """
        if bool(want_to_double):
            self._bet_multiplier += 1
            self._game.double_focus()
        self._game.move_focus()
        if not self._game.player_in_focus():
            self._game.move_focus(0)
            self._stage = 'HIT/STAND'

    def _hit_or_stand(self, want_to_hit: float) -> None:
        """ Hits or stands with the player in focus.

        Args:
            want_to_hit: A float representing whether the agent wants to hit
                with the player in focus.
        """
        if bool(want_to_hit):
            self._game.hit()
        else:
            self._game.stand()

    def _max_multiplier(self) -> int:
        """ Returns the number of times the player can split or double.

        The number of times the player can split or double without going
        bankrupt. When keept track of this information, the agent will not be
        able to achieve a negative bankroll.
        """
        return int(self._chips/self._bet_size)

    def _can_bet_more(self) -> bool:
        """ Return whether the agent is allowed to double down or split. """
        return self._max_multiplier() > self._bet_multiplier

    def _can_split(self) -> bool:
        """ Returns whether the agent is allowed to split. """
        return self._game.can_split() and self._can_bet_more()
