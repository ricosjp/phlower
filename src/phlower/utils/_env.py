import os


def determine_max_process(max_process: int | None = None) -> int:
    """Determine maximum number of processes.

    Parameters
    ----------
    max_process: int, optional
        Input maximum process.

    Returns
    -------
    resultant_max_process: int
    """
    if hasattr(os, "sched_getaffinity"):
        # This is more accurate in the cluster
        available_max_process = len(os.sched_getaffinity(0))
    else:
        available_max_process = os.cpu_count()

    if max_process is None:
        return available_max_process

    assert available_max_process is not None
    return min(available_max_process, max_process)
