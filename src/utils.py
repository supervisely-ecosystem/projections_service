import asyncio
from functools import wraps
from time import perf_counter
from typing import Callable, Set
from uuid import uuid4

import yaml

from supervisely.sly_logger import logger


def _log_execution_time(function_name: str, execution_time: float) -> None:
    """Log the execution time of the function.

    :param function_name: Name of the function.
    :type function_name: str
    :param execution_time: Execution time of the function.
    :type execution_time: float
    """
    logger.debug(f"{execution_time:.4f} sec | {function_name}")


def timeit(func: Callable) -> Callable:
    """Decorator to measure the execution time of the function.
    Works with both async and sync functions.

    :param func: Function to measure the execution time of.
    :type func: Callable
    :return: Decorated function.
    :rtype: Callable
    """

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = await func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            logger.debug(f"{execution_time:.4f} sec | {func.__name__}")
            return result

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = perf_counter()
            result = func(*args, **kwargs)
            end_time = perf_counter()
            execution_time = end_time - start_time
            _log_execution_time(func.__name__, execution_time)
            return result

        return sync_wrapper


def with_retries(retries: int = 3, sleep_time: int = 1, on_failure: Callable = None) -> Callable:
    """Decorator to retry the function in case of an exception.
    Works only with async functions. Custom function can be executed on failure.
    NOTE: The on_failure function should be idempotent and synchronous.

    :param retries: Number of retries.
    :type retries: int
    :param sleep_time: Time to sleep between retries.
    :type sleep_time: int
    :param on_failure: Function to execute on failure, if None, raise an exception.
    :type on_failure: Callable, optional
    :raises Exception: If the function fails after all retries.
    :return: Decorator.
    :rtype: Callable
    """

    def retry_decorator(func):
        @wraps(func)
        async def async_function_with_retries(*args, **kwargs):
            for _ in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.debug(f"Failed to execute {func.__name__}, retrying. Error: {str(e)}")
                    await asyncio.sleep(sleep_time)
            if on_failure is not None:
                return on_failure()
            else:
                raise Exception(f"Failed to execute {func.__name__} after {retries} retries.")

        return async_function_with_retries

    return retry_decorator


def read_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def track_in_progress_tasks(tasks: Set[str]) -> Callable:
    """Decorator to add task ID to in_progress_tasks set and remove it after the function execution.

    :param tasks: Set of in-progress task IDs.
    :type tasks: Set[str]
    :return: Decorator.
    :rtype: Callable
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                task_id = str(uuid4())
                tasks.add(task_id)
                try:
                    return await func(*args, **kwargs)
                finally:
                    tasks.remove(task_id)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                task_id = str(uuid4())
                tasks.add(task_id)
                try:
                    return func(*args, **kwargs)
                finally:
                    tasks.remove(task_id)

            return sync_wrapper

    return decorator
