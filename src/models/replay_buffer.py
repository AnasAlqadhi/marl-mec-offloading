"""
models/replay_buffer.py – Experience Replay Buffer
===================================================
Fixed-size circular buffer used by MADDPG.
Stores (state, action, reward, next_state) tuples.
"""

import random
from typing import List, Any


class ReplayBuffer:
    """Ring-buffer experience replay.

    Parameters
    ----------
    max_size : int
        Maximum number of transitions to keep.
    """

    def __init__(self, max_size: int = 200_000):
        self.buf: List[Any] = []
        self.max = max_size

    def add(self, transition) -> None:
        """Append a transition; evict oldest if over capacity."""
        self.buf.append(transition)
        if len(self.buf) > self.max:
            self.buf = self.buf[-self.max:]

    def sample(self, n: int) -> List[Any]:
        """Uniformly sample *n* transitions."""
        return random.sample(self.buf, n)

    def __len__(self) -> int:
        return len(self.buf)
