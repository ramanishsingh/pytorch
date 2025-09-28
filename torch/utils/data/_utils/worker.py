# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import os
import queue
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING, TypeVar, Union

import torch
from torch._utils import ExceptionWrapper

from . import HAS_NUMPY, IS_WINDOWS, MP_STATUS_CHECK_INTERVAL, signal_handling

from .stateful import Stateful


if TYPE_CHECKING:
    from torch.utils.data import Dataset

T = TypeVar("T")


# Stateful functionality
def try_to_serialize(obj: Any) -> Union[dict, None]:
    """Try to serialize an object if it implements Stateful protocol."""
    if isinstance(obj, Stateful):
        return obj.state_dict()
    return None


def try_to_deserialize(obj: T, state_dict: dict) -> T:
    """Try to deserialize an object if it implements Stateful protocol."""
    if isinstance(obj, Stateful) and state_dict is not None:
        obj.load_state_dict(state_dict)
    return obj


# Add alias for backward compatibility
_try_to_deserialize = try_to_deserialize


@dataclass(frozen=True)
class _AckStartup:
    """Dummy class used to ack startup and return state at time 0"""

    worker_id: int
    initial_state: Optional[Union[Dict[str, Any], ExceptionWrapper]]
    is_delta: bool = False


# State constants for stateful workers
_DATASET_ITER_STATE = "_dataset_iter_state"
_DATASET_STATE = "_dataset_state"
_FETCHER_ENDED = "_fetcher_ended"
_FETCHER_STATE = "_fetcher_state"
_WORKER_ID = "_worker_id"


class _IncrementalWorkerState:
    """Manages incremental state changes for worker processes."""

    def __init__(self, initial_worker_state_dict: Optional[Dict[str, Any]]):
        self._worker_id = None
        self._fetcher_ended = None

        dataset_state = None
        fetcher_iter_state = None
        if initial_worker_state_dict:
            self._worker_id = initial_worker_state_dict[_WORKER_ID]
            dataset_state = initial_worker_state_dict.get(_DATASET_STATE, None)
            fetcher_state = initial_worker_state_dict.get(_FETCHER_STATE, None)
            if fetcher_state is not None:
                self._fetcher_ended = fetcher_state[_FETCHER_ENDED]
                fetcher_iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)

        # Use simple state management for now (can be enhanced later with proper delta compression)
        self._dataset_state = dataset_state
        self._fetcher_iter_state = fetcher_iter_state

    def generate_delta(self, new_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        assert _WORKER_ID in new_state_dict
        self._worker_id = new_state_dict[_WORKER_ID]

        # For now, we return the full delta state
        # TODO: Implement proper delta compression like the reference
        incr_state_dict = {_WORKER_ID: self._worker_id, _FETCHER_STATE: None}

        ds_state = new_state_dict.get(_DATASET_STATE, None)
        if ds_state is not None:
            incr_state_dict[_DATASET_STATE] = ds_state
            self._dataset_state = ds_state

        fetcher_state = new_state_dict.get(_FETCHER_STATE, None)
        if fetcher_state is not None:
            self._fetcher_ended = fetcher_state[_FETCHER_ENDED]

            iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)
            if iter_state is not None:
                self._fetcher_iter_state = iter_state

            incr_state_dict[_FETCHER_STATE] = {
                _DATASET_ITER_STATE: iter_state,
                _FETCHER_ENDED: self._fetcher_ended,
            }
        return incr_state_dict

    def apply_delta(self, delta_state_dict: Dict[str, Any]) -> None:
        """Apply a delta to the current state."""
        self._worker_id = delta_state_dict[_WORKER_ID]
        ds_state = delta_state_dict.get(_DATASET_STATE, None)
        if ds_state is not None:
            self._dataset_state = ds_state

        fetcher_state = delta_state_dict.get(_FETCHER_STATE, None)
        if fetcher_state is not None:
            self._fetcher_ended = fetcher_state[_FETCHER_ENDED]
            iter_state = fetcher_state.get(_DATASET_ITER_STATE, None)
            if iter_state is not None:
                self._fetcher_iter_state = iter_state

    def get_state(self) -> Dict[str, Any]:
        """Get the current state."""
        fetcher_state = (
            {
                _FETCHER_ENDED: self._fetcher_ended,
                _DATASET_ITER_STATE: self._fetcher_iter_state,
            }
            if self._fetcher_ended is not None
            else None
        )
        return {
            _WORKER_ID: self._worker_id,
            _DATASET_STATE: self._dataset_state,
            _FETCHER_STATE: fetcher_state,
        }


if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import BOOL, DWORD, HANDLE

    # On Windows, the parent ID of the worker process remains unchanged when the manager process
    # is gone, and the only way to check it through OS is to let the worker have a process handle
    # of the manager and ask if the process status has changed.
    class ManagerWatchdog:
        def __init__(self) -> None:
            self.manager_pid = os.getppid()

            # mypy cannot detect this code is windows only
            self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD

            # Value obtained from https://msdn.microsoft.com/en-us/library/ms684880.aspx
            SYNCHRONIZE = 0x00100000
            self.manager_handle = self.kernel32.OpenProcess(
                SYNCHRONIZE, 0, self.manager_pid
            )

            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]

            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                # Value obtained from https://msdn.microsoft.com/en-us/library/windows/desktop/ms687032.aspx
                self.manager_dead = (
                    self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
                )
            return not self.manager_dead

else:

    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


_worker_info: Optional["WorkerInfo"] = None


class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: "Dataset"
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(
                f"Cannot assign attributes to {self.__class__.__name__} objects"
            )
        return super().__setattr__(key, val)

    def __repr__(self):
        items = [f"{k}={getattr(self, k)}" for k in self.__keys]
        return f"{self.__class__.__name__}({', '.join(items)})"


def get_worker_info() -> Optional[WorkerInfo]:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    return _worker_info


r"""Dummy class used to signal the end of an IterableDataset"""


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int


r"""Dummy class used to resume the fetching when worker reuse is enabled"""


@dataclass(frozen=True)
class _ResumeIteration:
    seed: Optional[int] = None


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# It's MIT licensed, here is the copyright:

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This function generates an array of int32 as the seed for
# `numpy.random`, in order to prevent state collision due to same
# seed and algorithm for `numpy.random` and `random` modules.
# TODO: Implement `SeedSequence` like object for `torch.random`
def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def _setup_worker_env(dataset, base_seed, worker_id, num_workers, shared_seed):
    """Common worker initialization for both stateless and stateful loops."""
    # Initialize C side signal handlers for SIGBUS and SIGSEGV.
    signal_handling._set_worker_signal_handlers()

    # Name thread and constrain torch threads
    torch.multiprocessing._set_thread_name("pt_data_worker")
    torch.set_num_threads(1)

    # Seed Python, Torch, and NumPy (if available)
    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np

        np.random.seed(np_seed)

    # Apply shared RNG for IterDataPipe graphs
    from torch.utils.data import IterDataPipe
    from torch.utils.data.graph_settings import apply_random_seed

    shared_rng = torch.Generator()
    if isinstance(dataset, IterDataPipe):
        assert shared_seed is not None
        shared_rng.manual_seed(shared_seed)
        dataset = apply_random_seed(dataset, shared_rng)

    # Populate global worker info
    global _worker_info
    _worker_info = WorkerInfo(
        id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset
    )

    return dataset, shared_rng, seed


# Sentinel object to distinguish between "not passed" and "passed as None"
_WORKER_LOOP_NON_STATEFUL_SENTINEL = object()


def _worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
    worker_state=_WORKER_LOOP_NON_STATEFUL_SENTINEL,
):
    """Unified worker loop for both stateful and non-stateful DataLoaders.

    Args:
        worker_state: Worker state for stateful DataLoader. When explicitly passed
                     (even if None), we operate in stateful mode. When the default
                     sentinel value is used, we operate in non-stateful mode.
    """
    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.

    # Determine if this is a stateful worker by checking if worker_state was passed
    is_stateful = worker_state is not _WORKER_LOOP_NON_STATEFUL_SENTINEL

    try:
        dataset, shared_rng, seed = _setup_worker_env(
            dataset, base_seed, worker_id, num_workers, shared_seed
        )
        from torch.utils.data import _DatasetKind, IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        # Initialize state management for stateful workers
        incremental_worker_state: Optional[_IncrementalWorkerState] = None
        initial_state = None
        is_delta = False

        init_exception = None
        fetcher = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            if is_stateful:
                # Stateful worker initialization
                if worker_state is None:
                    fetcher = _DatasetKind.create_fetcher(
                        dataset_kind, dataset, auto_collation, collate_fn, drop_last
                    )
                    initial_state = _make_state_dict(
                        worker_id, dataset_kind, fetcher, dataset
                    )
                    incremental_worker_state = _IncrementalWorkerState(initial_state)
                else:
                    # Always restore in this order:
                    #  1. try to restore dataset state
                    #  2. generate dataset iterator
                    #  3. try to restore iterator state
                    incremental_worker_state = _IncrementalWorkerState(worker_state)
                    if worker_state[_DATASET_STATE] is not None:
                        dataset = try_to_deserialize(
                            dataset, worker_state[_DATASET_STATE]
                        )
                    fetcher = _DatasetKind.create_fetcher(
                        dataset_kind, dataset, auto_collation, collate_fn, drop_last
                    )
                    if worker_state[_FETCHER_STATE] is not None:
                        if dataset_kind == _DatasetKind.Iterable:
                            if (
                                worker_state[_FETCHER_STATE][_DATASET_ITER_STATE]
                                is not None
                            ):
                                dataset_iter = try_to_deserialize(
                                    fetcher.dataset_iter,
                                    worker_state[_FETCHER_STATE][_DATASET_ITER_STATE],
                                )
                                if dataset_iter is not None:
                                    fetcher.dataset_iter = dataset_iter
                            # We always force fetcher to request at least one batch even if
                            # we know it will lead to immediate stop iteration
                            fetcher.ended = False
                    initial_state = incremental_worker_state.generate_delta(
                        _make_state_dict(worker_id, dataset_kind, fetcher, dataset)
                    )
                    is_delta = True
            else:
                # Non-stateful worker initialization (original behavior)
                fetcher = _DatasetKind.create_fetcher(
                    dataset_kind, dataset, auto_collation, collate_fn, drop_last
                )
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader worker process {worker_id}"
            )

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if is_stateful and isinstance(r, _AckStartup):
                # Stateful: Send ack and initial state to the main process
                data_queue.put(
                    (
                        r,
                        _AckStartup(
                            worker_id=worker_id,
                            initial_state=init_exception or initial_state,
                            is_delta=is_delta,
                        ),
                    )
                )
                del initial_state
                del is_delta
                continue
            elif isinstance(r, _ResumeIteration):
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                try:
                    # Recreate the fetcher for worker-reuse policy
                    fetcher = _DatasetKind.create_fetcher(
                        dataset_kind, dataset, auto_collation, collate_fn, drop_last
                    )
                    if is_stateful:
                        # see NOTE [ Incremental Worker State ]
                        initial_state = _make_state_dict(
                            worker_id, dataset_kind, fetcher, dataset
                        )
                        incremental_worker_state = _IncrementalWorkerState(
                            initial_state
                        )
                except Exception:
                    init_exception = ExceptionWrapper(
                        where=f"in DataLoader worker process {worker_id}"
                    )

                if is_stateful:
                    # Stateful: Acknowledge the main process with initial state
                    data_queue.put(
                        (
                            r,
                            _AckStartup(
                                worker_id=worker_id,
                                initial_state=init_exception or initial_state,
                            ),
                        )
                    )
                    del initial_state
                else:
                    # Non-stateful: Simple acknowledgment
                    data_queue.put((r, None))
                continue
            elif r is None:
                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue

            # Parse message format based on stateful mode
            if is_stateful:
                idx, (index, snapshot) = r
            else:
                idx, index = r

            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            delta_state_dict = None

            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    try:
                        data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                    except StopIteration:
                        if not dataset_kind == _DatasetKind.Iterable:
                            raise
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True

                    # Generate state delta for stateful workers when needed
                    if is_stateful and (snapshot or iteration_end):
                        # Generate incremental diff from prev_state_dict and current_state_dict
                        state_dict = _make_state_dict(
                            worker_id, dataset_kind, fetcher, dataset
                        )
                        delta_state_dict = incremental_worker_state.generate_delta(
                            state_dict
                        )
                        del state_dict
                except Exception as e:
                    if (
                        not is_stateful
                        and isinstance(e, StopIteration)
                        and dataset_kind == _DatasetKind.Iterable
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(
                            where=f"in DataLoader worker process {worker_id}"
                        )

            # Send data back with format based on stateful mode
            if is_stateful:
                data_queue.put((idx, (data, worker_id, delta_state_dict)))
                del data, idx, index, r, delta_state_dict  # save memory
            else:
                data_queue.put((idx, data))
                del data, idx, index, r  # save memory

    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()


def _make_state_dict(worker_id, dataset_kind, fetcher, dataset) -> Dict[str, Any]:
    """Create a state dictionary for the current worker state."""
    from torch.utils.data import _DatasetKind

    if dataset_kind == _DatasetKind.Iterable:
        fetcher_state = {
            _DATASET_ITER_STATE: try_to_serialize(fetcher.dataset_iter),
            _FETCHER_ENDED: fetcher.ended,
        }
        dataset_state = None
        if fetcher.dataset_iter is not fetcher.dataset:
            dataset_state = try_to_serialize(fetcher.dataset)
    else:
        fetcher_state = None
        # Pick up any user-defined dataset state
        dataset_state = try_to_serialize(dataset)

    return {
        _WORKER_ID: worker_id,
        _FETCHER_STATE: fetcher_state,
        _DATASET_STATE: dataset_state,
    }
