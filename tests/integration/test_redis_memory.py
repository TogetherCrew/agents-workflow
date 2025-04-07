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
        # Create instances with different keys
        self.memory1 = RedisMemory("test_key1")
        self.memory2 = RedisMemory("test_key2")
        self.memory3 = RedisMemory("test_key3")

        # Clean up any existing test keys
        self.memory1.delete_text()
        self.memory2.delete_text()
        self.memory3.delete_text()

    def tearDown(self):
        """Clean up after each test"""
        # Clean up test keys
        self.memory1.delete_text()
        self.memory2.delete_text()
        self.memory3.delete_text()

    def test_append_and_get_text(self):
        """Test appending and retrieving text from Redis"""
        # Test initial append
        success = self.memory1.append_text("Hello ")
        self.assertTrue(success)

        # Verify the text was stored
        result = self.memory1.get_text()
        self.assertEqual(result, "Hello ")

        # Test appending more text
        success = self.memory1.append_text("World!")
        self.assertTrue(success)

        # Verify the combined text
        result = self.memory1.get_text()
        self.assertEqual(result, "Hello World!")

    def test_multiple_keys(self):
        """Test handling multiple keys simultaneously"""
        # Store different values in different keys
        self.memory1.append_text("First value")
        self.memory2.append_text("Second value")

        # Verify each key has correct value
        self.assertEqual(self.memory1.get_text(), "First value")
        self.assertEqual(self.memory2.get_text(), "Second value")

    def test_delete_text(self):
        """Test deleting text from Redis"""
        # Store some text
        self.memory1.append_text("Delete me")

        # Verify text exists
        self.assertIsNotNone(self.memory1.get_text())

        # Delete the text
        success = self.memory1.delete_text()
        self.assertTrue(success)

        # Verify text is gone
        self.assertIsNone(self.memory1.get_text())

    def test_nonexistent_key(self):
        """Test behavior with nonexistent keys"""
        # Try to get nonexistent key
        result = self.memory1.get_text()
        self.assertIsNone(result)

        # Try to delete nonexistent key
        success = self.memory1.delete_text()
        self.assertFalse(success)

    def test_expiration(self):
        """Test that keys expire after the timeout period"""
        # Override timeout for testing (2 seconds instead of 15 minutes)
        original_timeout = self.memory3.timeout
        self.memory3.timeout = 2

        try:
            # Store some text
            self.memory3.append_text("Temporary text")

            # Verify text exists
            self.assertEqual(self.memory3.get_text(), "Temporary text")

            # Wait for expiration
            time.sleep(3)

            # Verify text is gone
            self.assertIsNone(self.memory3.get_text())

        finally:
            # Restore original timeout
            self.memory3.timeout = original_timeout

    def test_append_to_expired_key(self):
        """Test appending text to a key that has expired"""
        # Override timeout for testing
        original_timeout = self.memory3.timeout
        self.memory3.timeout = 2

        try:
            # Store initial text
            self.memory3.append_text("Initial ")

            # Wait for expiration
            time.sleep(3)

            # Append to expired key
            success = self.memory3.append_text("New text")
            self.assertTrue(success)

            # Verify only new text exists
            result = self.memory3.get_text()
            self.assertEqual(result, "New text")

        finally:
            # Restore original timeout
            self.memory3.timeout = original_timeout
