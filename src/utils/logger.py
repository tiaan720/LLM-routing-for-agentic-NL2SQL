import logging
import os

import dotenv

dotenv.load_dotenv()


def configure_logging(level: str = "INFO") -> None:
    """Configure logging with the specified level.

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Suppress INFO logs from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)


configure_logging(os.getenv("LOGGING_LEVEL", "INFO"))


logger = logging.getLogger(__name__)
