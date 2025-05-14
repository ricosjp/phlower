import concurrent.futures as cf
import multiprocessing as mp
from collections.abc import Callable, Iterable
from functools import partial
from logging import getLogger
from typing import Any, TypeVar

from phlower.utils import determine_max_process
from phlower.utils.exceptions import PhlowerMultiProcessError

T1 = TypeVar("T1")
T2 = TypeVar("T2")

_logger = getLogger(__name__)


def _process_chunk(fn: Callable[[T1], T2], chunk: list[tuple[Any]]) -> list[T1]:
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _get_chunks(
    iterables: Iterable[T1 | tuple[T1]], chunksize: int
) -> list[tuple[T1]]:
    """Iterates over ziped iterables in chunks."""
    iterables = _format_inputs(iterables)
    return [
        tuple(iterables[i : i + chunksize])
        for i in range(0, len(iterables), chunksize)
    ]


def _format_inputs(inputs: list) -> list[tuple[T1]]:
    assert len(inputs) > 0

    if isinstance(inputs[0], tuple):
        return inputs

    return [(v,) for v in inputs]


def _santize_futures(futures: list[cf.Future]) -> None:
    for future in futures:
        ex = future.exception()
        if ex is None:
            continue

        raise PhlowerMultiProcessError(
            "Some jobs are failed during multiprocess execution. "
            "If content of exception shown above is only a integer number "
            "such as '1', it means that child process is killed by host system"
            " like OOM killer."
        ) from ex


class PhlowerMultiprocessor:
    def __init__(self, max_process: int | None):
        self._max_process = determine_max_process(max_process)

    def get_determined_process(self) -> int:
        return self._max_process

    def run(
        self,
        inputs: list[tuple[T1] | T1],
        target_fn: Callable[[T1], T2],
        chunksize: int | None = None,
        mp_context: str | None = None,
    ) -> list[T2]:
        """Wrapper function for concurrent.futures
         to run safely with multiple processes.

        Parameters
        ----------
        inputs: list[tuple[T1]]
            List of arguments which are supposed to be run in parallel
            Each item of list corresponds to packed arguments for target_fn
        max_process : int
            the number of processes to use
        target_fn : Callable[[T1], T2]
            function to execute
        chunksize : int, optional
            chunck size. By default, automatically determined.

        Returns
        -------
        list[T2]
            Iterable of objects returned from target_fn

        Raises
        -------
        PhlowerMultiprocessError:
            If some processes are killed by host system such as OOM killer,
             this error raises.
        """
        if len(inputs) == 0:
            return []

        if self._max_process == 1:
            # NOTE: This is a workaround to avoid pickling objects
            # when max_process is 1.
            return [target_fn(*arg) for arg in _format_inputs(inputs)]

        futures: list[cf.Future] = []
        chunksize = chunksize or self._determine_chunksize(len(inputs))
        _logger.info(f"chunksize is set as {chunksize}.")

        with cf.ProcessPoolExecutor(
            self._max_process, mp_context=mp.get_context(mp_context)
        ) as executor:
            for chunk in _get_chunks(inputs, chunksize=chunksize):
                future = executor.submit(
                    partial(_process_chunk, target_fn), chunk
                )

                futures.append(future)

            cf.wait(futures)

        _santize_futures(futures)

        # flatten
        return sum([f.result() for f in futures], start=[])

    def _determine_chunksize(self, n_items: int) -> int:
        return int(
            max(
                n_items // determine_max_process(),
                1,
            )
        )
