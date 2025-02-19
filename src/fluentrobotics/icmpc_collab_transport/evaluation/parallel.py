"""
This module contains helper utilities to execute tasks in parallel while
coordinating loguru and tqdm output.

This module uses the multiprocessing library to enable parallelism. Thus, it
really only makes sense to use these utilities for very compute-heavy tasks
since there is a large communication overhead between processes. Furthermore,
only pickleable objects can be used as function parameters and return values.

Some use cases may be:

- Processing bagfiles into training datasets.
  - Transferring a 1GB numpy array between processes (which involves pickling
    and unpickling) takes a few seconds, but generating the dataset can take
    several minutes.
- Processing videos from bagfiles.

Example Usage
>>> def do_something(filename: str) -> None:
...     # Child processes using tqdm must set the position kwarg
...     for ... in tqdm.tqdm(position=parallel.tqdm_position(), ...):
...         ...

>>> def add(a: int, b: int) -> int:
...     return a + b

>>> bagfiles = ["a.bag", "b.bag", "c.bag"]
>>> parallel.fork_join(do_something, bagfiles, 3)
[None, None, None]

>>> # The returned list will be in arbitrary order.
>>> parallel.fork_join(add, [(1, 2), (3, 4), (5, 6)])
[11, 3, 7]
"""

import concurrent.futures
import multiprocessing
import multiprocessing.managers
import multiprocessing.synchronize
import os
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    ClassVar,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tqdm
from loguru import logger


class _Internal:
    """Private namespace scope for this module."""

    executor_lock: ClassVar[multiprocessing.synchronize.Lock] = multiprocessing.Lock()
    executor: ClassVar[Optional[concurrent.futures.ProcessPoolExecutor]] = None
    executor_nprocs: ClassVar[int] = 0
    manager: ClassVar[multiprocessing.managers.SyncManager] = multiprocessing.Manager()
    """Manager to handle shared state between processes. Assumed to have child
    process id #1."""

    @classmethod
    @contextmanager
    def get_executor(
        cls, n_procs: int
    ) -> Generator[concurrent.futures.ProcessPoolExecutor, None, None]:
        """Returns the internal ProcessPoolExecutor wrapped in a context
        manager.

        Manages the cleanup of worker processes in the event that the user
        requests a SIGINT (Ctrl-C). Only one thread can use the Executor at a
        time.

        Args:
            n_procs (int): The number of worker processes used to initialize
                the Executor. This argument is ignored if the Executor has
                already been initialized.

        Example:
            with _Internal.get_executor(5) as executor:
                ...
        """
        if cls.executor is None:
            logger.remove()
            logger.add(
                lambda msg: tqdm.tqdm.write(msg, end=""),  # type: ignore
                colorize=True,
                enqueue=True,
            )
            tqdm.tqdm.set_lock(multiprocessing.RLock())
            cls.executor = concurrent.futures.ProcessPoolExecutor(n_procs)
            cls.executor_nprocs = n_procs

        with cls.executor_lock:
            try:
                yield cls.executor
            except KeyboardInterrupt:
                for pid in cls.executor._processes.keys():
                    os.kill(pid, signal.SIGTERM)
            finally:
                cls.executor.shutdown()
                cls.executor = None

    @staticmethod
    def idle_bar(stop_event: threading.Event) -> None:
        """Target function that spins an idle progress bar.

        When a progress bar finishes in a concurrent environment it no longer
        updates line it occupied. When logging messages are output to the
        screen, all progress bars are shifted by one line. This function
        continuously updates a line to keep the output tidy.

        Args:
            stop_event (Event): Signals when this task should terminate.
        """
        position = tqdm_position()
        pbar = tqdm.tqdm(
            desc=f"({position}) idle",
            position=position,
            dynamic_ncols=True,
            total=1,
            leave=False,
        )
        while not stop_event.is_set():
            pbar.update(0)
            time.sleep(0.25)
        pbar.close()


def tqdm_position() -> int:
    """Returns the tqdm position of this process in a multiprocessing
    environment.

    Assumes only a single manager and a single pool are spawned.
    """
    if multiprocessing.parent_process() is None:
        return 0
    else:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Process.name
        # The manager process has id 1
        return int(multiprocessing.current_process().name.split("-")[1]) - 1


class _TryTask:
    def __init__(self, task: Callable[..., Any], args: Any):
        self._task = task
        self._args = args

    def __call__(self) -> Any:
        try:
            if self._args is None:
                return self._task()
            elif type(self._args) is tuple:
                return self._task(*self._args)
            else:
                return self._task(self._args)
        except BaseException as ex:
            logger.error(
                f"""{ex}
{"".join(traceback.format_tb(ex.__traceback__)[1:]).rstrip()}
  Task: {self._task}
  Args: {self._args}"""
            )


def fork_join(
    task: Callable[..., Any],
    task_args: Sequence[Union[Any, Tuple[Any, ...]]],
    n_proc: int = 2,
) -> List[Any]:
    """Parallelize calls to a function using pooled fork-join.

    Args:
        task: A callable function object.
        task_args: A list of arguments for each call to the function.
            Each element of this list may be None (if the function does
            not take any arguments), an object, or a tuple of objects.
            Note that there is no distinction between None as a signaling
            mechanism and None as an intentional parameter.
        n_proc: The number of processes to use. If a pool has already been created
            this argument will be ignored and the existing pool will be reused.

    Returns:
        A list containing the return values of calls to the task function in
        arbitrary order.
    """
    with _Internal.get_executor(n_proc) as executor:
        task_futures: List[concurrent.futures.Future[Any]] = []
        for args in task_args:
            task_futures.append(executor.submit(_TryTask(task, args)))

        stop_event = _Internal.manager.Event()
        idle_futures: List[concurrent.futures.Future[None]] = []
        for _ in range(_Internal.executor_nprocs):
            idle_futures.append(executor.submit(_Internal.idle_bar, stop_event))

        results: List[Any] = []

        fut: concurrent.futures.Future[Any]
        for fut in tqdm.tqdm(
            concurrent.futures.as_completed(task_futures),
            position=0,
            leave=False,
            total=len(task_futures),
        ):
            results.append(fut.result())

        stop_event.set()
        concurrent.futures.wait(idle_futures)
        # Clears the contents of the line if tqdm did not clean up properly
        print(end="\33[2K\r")
        return results
