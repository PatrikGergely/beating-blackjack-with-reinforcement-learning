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

"""Utilities for logging to a file."""

import logging


def create_logger(filename: str) -> logging.Logger:
    """ Returns a logger that logs INFO messages to a file.

    Creates a logger that logs messages with level INFO or higher to a file
    and logs messages with level WARNING or higher to the standard stream.

    Args:
        filename: The name of the file to log to.
    """
    logger = logging.getLogger(filename)
    logger.propagate=False
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '+
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
