import concurrent.futures as cf
import itertools
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, TypeVar

from phlower.utils import determine_max_process
from phlower.utils.exceptions import PhlowerMultiProcessError

T = TypeVar("T")


def _process_chunk(
    fn: Callable[[Any], T], chunk: list[Iterable[Any]]
) -> list[T]:
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _get_chunks(
    *iterables: Iterable[Any], chunksize: int
) -> Iterable[list[Any]]:
    """Iterates over ziped iterables in chunks."""
    it = zip(*iterables, strict=False)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


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
        self.max_process = max_process

    def run(
        self,
        *inputs: list[Any],
        target_fn: Callable[[Any], T],
        chunksize: int | None = None,
    ) -> list[T]:
        """Wrapper function for concurrent.futures
         to run safely with multiple processes.

        Parameters
        ----------
        max_process : int
            the number of processes to use
        target_fn : Callable[[Any], T]
            function to execute
        chunksize : int, optional
            chunck size, by default 1

        Returns
        -------
        list[T]
            Iterable of objects returned from target_fn

        Raises
        -------
        PhlowerMultiprocessError:
            If some processes are killed by host system such as OOM killer,
             this error raises.
        """
        futures: list[cf.Future] = []

        chunksize = chunksize or self._determine_chunksize()

        with cf.ProcessPoolExecutor(self.max_process) as executor:
            for chunk in _get_chunks(*inputs, chunksize=chunksize):
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
