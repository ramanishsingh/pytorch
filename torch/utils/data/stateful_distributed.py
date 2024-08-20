# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Iterator, Optional, Sized

import torch
from . import Sampler, Dataset
from .stateful import Stateful


class _StatefulDistributedSamplerIterator(Iterator[int], Stateful):

    def __init__(self, sampler, parent_iterator: Iterator[int]):
        self.sampler = sampler
        self.parent_iterator = parent_iterator

    def __next__(self) -> int:
        if self.sampler.next_yielded is not None:
            for _ in range(self.sampler.next_yielded):
                next(self.parent_iterator)

            self.sampler.yielded = self.sampler.next_yielded
            self.sampler.next_yielded = None

        val = next(self.parent_iterator)
        self.sampler.yielded += 1
        return val


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:

        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        return _StatefulDistributedSamplerIterator(self, super().__iter__())

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.next_yielded = state_dict[self._YIELDED]
