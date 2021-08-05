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
""" Implements a single-process DQN agent based Bettor. """

from typing import List, Tuple

from acme import specs
from acme.agents.tf.dqn import DQN
from acme.tf import utils as tf2_utils
import sonnet as snt
import numpy as np

from bbwrl.bot.bettors.ml.trainer_bettor import TrainerBettor
from bbwrl.bot.bettors.ml.policy_bettor import PolicyBettor
from bbwrl.environments import rule_variation


# The possible bet sizes the DQN agent can choose from.
ACTIONS = (1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0)


def _environment_spec(actions: Tuple[float, ...]) -> specs.EnvironmentSpec:
    """ Generates an environment specification based on the action space. """
    return specs.EnvironmentSpec(
        observations={
            'CHIPS': specs.BoundedArray(
                (1,), np.float32, minimum=0.0, maximum=1000.0),
            'CARD_DISTRIBUTION': specs.BoundedArray(
                (14,), np.float32, minimum=0,
                maximum=4*rule_variation.SHOE_SIZE)
            },
        actions=specs.DiscreteArray(len(actions)),
        rewards=specs.BoundedArray((),
                                   np.float32,
                                   minimum=-100.0,
                                   maximum=100.0),
        discounts=specs.BoundedArray((),
                                     np.float32,
                                     minimum=0.0,
                                     maximum=1.0)
    )


def _network(environment_spec: specs.EnvironmentSpec,
             network_layer_sizes: List[int]) -> snt.Module:
    """ Returns a network.

    Args:
        environment_spec: The specification for the environment.
        network_layer_sizes: The size of each layer in the network.
    """
    return snt.Sequential([
          tf2_utils.batch_concat,
          snt.nets.MLP(network_layer_sizes +
                       [environment_spec.actions.num_values])
      ])


# pylint: disable=invalid-name
def DQNTrainer(network_shape: List[int],
               epsilon: float,
               learning_rate: float,
               batch_size: int) -> TrainerBettor:
    """ Returns a TrainerBettor that uses a DQN agent.

    Args:
        network_shape: The size of each layer in the network.
        epsilon: probability of taking a random action.
        learning_rate: learning rate for the q-network update.
        batch_size: batch size for updates.
    """
    batch_size = round(batch_size)
    env_spec = _environment_spec(ACTIONS)
    return TrainerBettor(DQN(environment_spec=env_spec,
                             network=_network(env_spec, network_shape),
                             learning_rate=learning_rate,
                             epsilon=epsilon,
                             batch_size=batch_size,
                             ),
                         lambda x: ACTIONS[x],
                         'float32')


def DQNBettor(policy_path: str) -> PolicyBettor:
    """ Returns a DQN PolicyBettor.

    Args:
        policy_path: Path to the saved model.
    """
    return PolicyBettor(policy_path,
                        lambda x: ACTIONS[np.argmax(x[0])],
                        lambda x: x,
                        'float32')
