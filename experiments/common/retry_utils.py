import time

from src.utils.logger import logger


def retry_with_backoff(fn, max_retries=3, initial_delay=1, *args, **kwargs):
    """
    Generic retry logic with exponential backoff.
    Args:
        fn: Function to call.
        max_retries: Maximum number of retries.
        initial_delay: Initial delay in seconds.
        *args, **kwargs: Arguments for fn.
    Returns:
        Result of fn(*args, **kwargs) or raises last exception.
    """
    retry_delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except (GeneratorExit, ConnectionError, TimeoutError, OSError) as e:
            if attempt < max_retries:
                logger.warning(
                    f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                logger.error(f"Failed after {max_retries + 1} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Non-retryable error: {e}")
            raise
