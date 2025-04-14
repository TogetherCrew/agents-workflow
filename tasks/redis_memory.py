import logging
import redis
from typing import Optional
import os
from datetime import timedelta


class RedisMemory:
    """A class for persisting text in Redis with a timeout.

    A Redis-based memory system that allows storing and retrieving text with an
    automatic expiration timeout.
    """

    def __init__(self, key: str):
        """Initialize Redis connection using environment variables.

        Parameters
        ----------
        key : str
            The Redis key to use for storing the text
        """
        self.redis_host = os.getenv("REDIS_HOST")
        self.redis_port = os.getenv("REDIS_PORT")
        self.redis_password = os.getenv("REDIS_PASSWORD")

        if not self.redis_host or not self.redis_port or not self.redis_password:
            raise ValueError("All REDIS_HOST, REDIS_PORT, and REDIS_PASSWORD must be set")

        self.key = key
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=int(self.redis_port),
            password=self.redis_password,
            decode_responses=True,  # Automatically decode responses to strings
        )

        # Set timeout
        self.timeout = timedelta(minutes=15).total_seconds()

    def append_text(self, text: str) -> bool:
        """Append text to the value stored at key. If key doesn't exist, create it.

        Parameters
        ----------
        text : str
            The text to append to the existing value or create as new value

        Returns
        -------
        bool
            True if operation was successful, False otherwise
        """
        try:
            # Get current value (if exists)
            current_value = self.redis_client.get(self.key)

            # If key exists, append to it; otherwise create new entry
            if current_value:
                new_value = current_value + text
                # Reset expiration time when updating
                success = self.redis_client.setex(
                    self.key, int(self.timeout), new_value
                )
            else:
                success = self.redis_client.setex(self.key, int(self.timeout), text)

            return success
        except Exception as e:
            logging.error(f"Error appending text to Redis: {e}")
            return False

    def get_text(self) -> Optional[str]:
        """Get the text stored at the given key.

        Returns
        -------
        str or None
            The text if key exists, None otherwise
        """
        try:
            return self.redis_client.get(self.key)
        except Exception as e:
            logging.error(f"Error getting text from Redis: {e}")
            return None

    def delete_text(self) -> bool:
        """Delete the text stored at the given key.

        Returns
        -------
        bool
            True if deleted successfully, False otherwise
        """
        try:
            return bool(self.redis_client.delete(self.key))
        except Exception as e:
            logging.error(f"Error deleting text from Redis: {e}")
            return False
