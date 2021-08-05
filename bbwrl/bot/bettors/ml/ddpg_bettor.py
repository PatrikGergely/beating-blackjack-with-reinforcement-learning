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
"""
Implements a Deep Deterministic Policy Gradient agent based Bettor.

When used with ray.tune this module needs to be used with float64, otherwise
with float32. (To tune, replace all float32 with float64 in this file)
"""

from typing import Any, List, Dict

from acme import specs
from acme.tf import networks
from acme.tf import utils as tf2_utils
import sonnet as snt
import numpy as np
import tensorflow as tf

from bbwrl.bot.bettors.ml.trainer_bettor import TrainerBettor
from bbwrl.bot.bettors.ml.policy_bettor import PolicyBettor
from bbwrl.environments import rule_variation
from bbwrl.utils.modified_ddpg import ModifiedDDPG


def _environment_spec() -> specs.EnvironmentSpec:
    """ Generates the environment specification. """
    return specs.EnvironmentSpec(
        observations={
            'CHIPS': specs.BoundedArray(
                (1,), np.float32, minimum=0.0, maximum=1000.0),
            'CARD_DISTRIBUTION': specs.BoundedArray(
                (14,), np.float32, minimum=0,
                maximum=4*rule_variation.SHOE_SIZE)
            },
        actions=specs.BoundedArray((),
                                   np.float32,
                                   minimum=1.0,
                                   maximum=60.0),
        rewards=specs.BoundedArray((),
                                   np.float32,
                                   minimum=-100.0,
                                   maximum=100.0),
        discounts=specs.BoundedArray((),
                                     np.float32,
                                     minimum=0.0,
                                     maximum=1.0)
    )


def _networks(
    action_spec: specs.BoundedArray,
    policy_layer_sizes: List[int],
    critic_layer_sizes: List[int],
) -> Dict[str, snt.Module]:
    """Creates networks used by the DDPG agent.

    Args:
        action_spec: The specification for the action space.
        policy_layer_sizes: The size of each layer in the policy network.
        critic_layer_sizes: The size of each layer in the critic network.

    Returns:
        A dictionary containing the policy, critic and observation network.
    """

    num_dimensions = np.prod(action_spec.shape, dtype=int)
    policy_layer_sizes = policy_layer_sizes + [num_dimensions]
    critic_layer_sizes = critic_layer_sizes + [1]

    policy_network = snt.Sequential([
        snt.nets.MLP(policy_layer_sizes),
        tf.tanh
    ])

    # The multiplexer concatenates the (maybe transformed) observations/actions.
    critic_network = networks.CriticMultiplexer(
        critic_network=snt.nets.MLP(critic_layer_sizes)
    )

    return {
        'policy': policy_network,
        'critic': critic_network,
        'observation': tf2_utils.batch_concat,
    }

# pylint: disable=invalid-name
def DDPGTrainer(policy_network_shape,
                critic_network_shape,
                policy_learning_rate,
                critic_learning_rate,
                batch_size,
                sigma) -> TrainerBettor:
    """ Returns a TrainerBettor that uses a DDPG agent.

    Args:
        policy_network_shape: The size of each layer in the policy network.
        critic_network_shape: The size of each layer in the critic network.
        policy_learning_rate: learning rate for the policy network update.
        critic_learning_rate: learning rate for the critic network update.
        batch_size: batch size for updates.
        sigma: standard deviation of zero-mean, Gaussian exploration noise.
    """
    batch_size = round(batch_size)
    env_spec = _environment_spec()
    network = _networks(env_spec.actions,
                        policy_network_shape,
                        critic_network_shape)
    return TrainerBettor(ModifiedDDPG(environment_spec=env_spec,
                              policy_network = network['policy'],
                              critic_network = network['critic'],
                              observation_network = network['observation'],
                              policy_learning_rate=policy_learning_rate,
                              critic_learning_rate=critic_learning_rate,
                              sigma=sigma,
                              batch_size=batch_size,
                              discount=1.0
                              ),
                         lambda x: max(1.0,(x+1.0)*29.5+0.9),
                         'float32')


def _convert2tensor(d: Dict[Any, float]) -> Dict[Any, tf.Tensor]:
    """ Converts a dictionary to a dictionary of tensors. """
    t_d={}
    for k in d:
        t_d[k] = tf.constant(d[k], dtype='float32')
    return t_d


def DDPGBettor(policy_path) -> PolicyBettor:
    """ Returns a DDPG PolicyBettor.

    Args:
        policy_path: Path to the saved model.
    """
    return PolicyBettor(policy_path,
                        lambda x: max(1.0,
                                      (np.asscalar(x.numpy())+1.0)*29.5+0.9),
                        lambda x: tf2_utils.batch_concat(_convert2tensor(x)),
                        'float32')
