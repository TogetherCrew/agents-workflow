import unittest
from unittest.mock import patch, MagicMock
import os
from tasks.redis_memory import RedisMemory


class TestRedisMemory(unittest.TestCase):
    """Test cases for the RedisMemory class"""

    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "REDIS_HOST": "test-host",
                "REDIS_PORT": "6379",
                "REDIS_PASSWORD": "test-password",
            },
        )
        self.env_patcher.start()

        # Mock the Redis client
        self.redis_client_mock = MagicMock()
        self.redis_patcher = patch("redis.Redis", return_value=self.redis_client_mock)
        self.redis_mock = self.redis_patcher.start()

        # Create instance of RedisMemory with mocked dependencies
        self.memory = RedisMemory(key='test_key')
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
        self.redis_patcher.stop()

    def test_init_with_env_vars(self):
        """Test initialization with environment variables"""
        self.redis_mock.assert_called_once_with(
            host="test-host", port=6379, password="test-password", decode_responses=True
        )
        self.assertEqual(self.memory.key, "test_key")

    def test_init_with_default_values(self):
        """Test initialization with default values when env vars are not set"""
        self.env_patcher.stop()
        # Clear existing redis client mock calls
        self.redis_mock.reset_mock()

        # Create new instance with no env vars
        memory = RedisMemory("test_key")

        self.redis_mock.assert_called_once_with(
            host="localhost", port=6379, password="", decode_responses=True
        )
        self.assertEqual(memory.key, "test_key")

    def test_append_text_new_key(self):
        """Test appending text to a new key"""
        # Mock Redis get to return None (key doesn't exist)
        self.redis_client_mock.get.return_value = None
        self.redis_client_mock.setex.return_value = True

        result = self.memory.append_text("test text")

        self.redis_client_mock.get.assert_called_once_with("test_key")
        self.redis_client_mock.setex.assert_called_once()
        self.assertEqual(self.redis_client_mock.setex.call_args[0][0], "test_key")
        self.assertEqual(self.redis_client_mock.setex.call_args[0][2], "test text")
        self.assertTrue(result)

    def test_append_text_existing_key(self):
        """Test appending text to an existing key"""
        # Mock Redis get to return an existing value
        self.redis_client_mock.get.return_value = "existing "
        self.redis_client_mock.setex.return_value = True

        result = self.memory.append_text("text")

        self.redis_client_mock.get.assert_called_once_with("test_key")
        self.redis_client_mock.setex.assert_called_once()
        self.assertEqual(self.redis_client_mock.setex.call_args[0][0], "test_key")
        self.assertEqual(self.redis_client_mock.setex.call_args[0][2], "existing text")
        self.assertTrue(result)

    def test_append_text_exception(self):
        """Test handling of exceptions in append_text"""
        # Mock Redis get to raise an exception
        self.redis_client_mock.get.side_effect = Exception("Test exception")

        result = self.memory.append_text("text")

        self.assertFalse(result)

    def test_get_text_existing_key(self):
        """Test getting text from an existing key"""
        # Mock Redis get to return a value
        self.redis_client_mock.get.return_value = "test value"

        result = self.memory.get_text()

        self.redis_client_mock.get.assert_called_once_with("test_key")
        self.assertEqual(result, "test value")

    def test_get_text_nonexistent_key(self):
        """Test getting text from a nonexistent key"""
        # Mock Redis get to return None
        self.redis_client_mock.get.return_value = None

        result = self.memory.get_text()

        self.redis_client_mock.get.assert_called_once_with("test_key")
        self.assertIsNone(result)

    def test_get_text_exception(self):
        """Test handling of exceptions in get_text"""
        # Mock Redis get to raise an exception
        self.redis_client_mock.get.side_effect = Exception("Test exception")

        result = self.memory.get_text()

        self.assertIsNone(result)

    def test_delete_text_success(self):
        """Test successful deletion of a key"""
        # Mock Redis delete to return 1 (successful deletion)
        self.redis_client_mock.delete.return_value = 1

        result = self.memory.delete_text()

        self.redis_client_mock.delete.assert_called_once_with("test_key")
        self.assertTrue(result)

    def test_delete_text_nonexistent_key(self):
        """Test deletion of a nonexistent key"""
        # Mock Redis delete to return 0 (key didn't exist)
        self.redis_client_mock.delete.return_value = 0

        result = self.memory.delete_text()

        self.redis_client_mock.delete.assert_called_once_with("test_key")
        self.assertFalse(result)

    def test_delete_text_exception(self):
        """Test handling of exceptions in delete_text"""
        # Mock Redis delete to raise an exception
        self.redis_client_mock.delete.side_effect = Exception("Test exception")

        result = self.memory.delete_text()

        self.assertFalse(result)
