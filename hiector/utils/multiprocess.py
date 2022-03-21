from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, List, Optional

from tqdm.auto import tqdm


def multiprocess(
    process_fun: Callable, arguments: Iterable[Any], total: Optional[int] = None, max_workers: int = 4
) -> List[Any]:
    """
    Executes multiprocessing with tqdm.
    Parameters
    ----------
    process_fun: A function that processes a single item.
    arguments: Arguments with which te function is called.
    total: Number of iterations to run (for cases of iterators)
    max_workers: Max workers for the process pool executor.

    Returns A list of results.
    -------


    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_fun, arguments), total=total))
    return results
