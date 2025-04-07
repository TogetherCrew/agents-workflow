import unittest
import os
from tasks.redis_memory import RedisMemory
import time


class TestRedisMemoryIntegration(unittest.TestCase):
    """Integration tests for RedisMemory class with actual Redis instance"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests"""
        # Ensure Redis environment variables are set for testing
        os.environ.setdefault("REDIS_HOST", "localhost")
        os.environ.setdefault("REDIS_PORT", "6379")
        os.environ.setdefault("REDIS_PASSWORD", "")

    def setUp(self):
        """Set up before each test"""
        self.memory = RedisMemory(key='test_key')
        # Clean up any existing test keys
        self.test_keys = ["test_key1", "test_key2", "test_key3"]
        for key in self.test_keys:
            self.memory.delete_text(key)

    def tearDown(self):
        """Clean up after each test"""
        # Clean up test keys
        for key in self.test_keys:
            self.memory.delete_text(key)

    def test_append_and_get_text(self):
        """Test appending and retrieving text from Redis"""
        key = "test_key1"

        # Test initial append
        success = self.memory.append_text(key, "Hello ")
        self.assertTrue(success)

        # Verify the text was stored
        result = self.memory.get_text(key)
        self.assertEqual(result, "Hello ")

        # Test appending more text
        success = self.memory.append_text(key, "World!")
        self.assertTrue(success)

        # Verify the combined text
        result = self.memory.get_text(key)
        self.assertEqual(result, "Hello World!")

    def test_multiple_keys(self):
        """Test handling multiple keys simultaneously"""
        # Store different values in different keys
        self.memory.append_text("test_key1", "First value")
        self.memory.append_text("test_key2", "Second value")

        # Verify each key has correct value
        self.assertEqual(self.memory.get_text("test_key1"), "First value")
        self.assertEqual(self.memory.get_text("test_key2"), "Second value")

    def test_delete_text(self):
        """Test deleting text from Redis"""
        key = "test_key1"

        # Store some text
        self.memory.append_text(key, "Delete me")

        # Verify text exists
        self.assertIsNotNone(self.memory.get_text(key))

        # Delete the text
        success = self.memory.delete_text(key)
        self.assertTrue(success)

        # Verify text is gone
        self.assertIsNone(self.memory.get_text(key))

    def test_nonexistent_key(self):
        """Test behavior with nonexistent keys"""
        # Try to get nonexistent key
        result = self.memory.get_text("nonexistent_key")
        self.assertIsNone(result)

        # Try to delete nonexistent key
        success = self.memory.delete_text("nonexistent_key")
        self.assertFalse(success)

    def test_expiration(self):
        """Test that keys expire after the timeout period"""
        key = "test_key3"

        # Override timeout for testing (2 seconds instead of 2 hours)
        original_timeout = self.memory.timeout
        self.memory.timeout = 2

        try:
            # Store some text
            self.memory.append_text(key, "Temporary text")

            # Verify text exists
            self.assertEqual(self.memory.get_text(key), "Temporary text")

            # Wait for expiration
            time.sleep(3)

            # Verify text is gone
            self.assertIsNone(self.memory.get_text(key))

        finally:
            # Restore original timeout
            self.memory.timeout = original_timeout

    def test_append_to_expired_key(self):
        """Test appending text to a key that has expired"""
        key = "test_key3"

        # Override timeout for testing
        original_timeout = self.memory.timeout
        self.memory.timeout = 2

        try:
            # Store initial text
            self.memory.append_text(key, "Initial ")

            # Wait for expiration
            time.sleep(3)

            # Append to expired key
            success = self.memory.append_text(key, "New text")
            self.assertTrue(success)

            # Verify only new text exists
            result = self.memory.get_text(key)
            self.assertEqual(result, "New text")

        finally:
            # Restore original timeout
            self.memory.timeout = original_timeout
