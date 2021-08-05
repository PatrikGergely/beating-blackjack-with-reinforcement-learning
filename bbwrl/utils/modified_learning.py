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

"""An extension of DDPGLearner that saves snapshots of the policy network."""

from acme.agents.tf.ddpg import learning
from acme.tf import savers as tf2_savers


class ModifiedDDPGLearner(learning.DDPGLearner):
    """An extension of DDPGLearner that saves snapshots of the policy network.

    After initializing the base class creates a snapshotter that saves the
    policy network every 60 minutes.
    """
    def __init__(self, *args, **kwargs):
        """See base class."""
        super().__init__(*args, **kwargs)
        self._snapshotter = tf2_savers.Snapshotter(
            objects_to_save={'network': self._policy_network},
            time_delta_minutes=60.)

    def step(self):
        """See base class."""
        super().step()
        self._snapshotter.save()
