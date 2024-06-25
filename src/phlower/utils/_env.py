import os


def determine_n_process(max_process: int | None) -> int:
    if max_process is None:
        return os.cpu_count()

    return min(max_process, os.cpu_count())
