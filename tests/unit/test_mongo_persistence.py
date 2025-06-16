import unittest
from unittest.mock import patch, MagicMock
import os
from datetime import datetime
from bson import ObjectId
from tasks.mongo_persistence import MongoPersistence


class TestMongoPersistence(unittest.TestCase):
    """Test cases for the MongoPersistence class"""

    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "MONGODB_HOST": "test-host",
                "MONGODB_PORT": "27017",
                "MONGODB_USER": "test-user",
                "MONGODB_PASS": "test-password",
            },
        )
        self.env_patcher.start()

        # Mock the MongoDB client and collection
        self.collection_mock = MagicMock()
        self.db_mock = MagicMock()
        self.db_mock.get_collection.return_value = self.collection_mock
        
        self.client_mock = MagicMock()
        self.client_mock.get_database.return_value = self.db_mock
        
        self.mongo_patcher = patch("pymongo.MongoClient", return_value=self.client_mock)
        self.mongo_mock = self.mongo_patcher.start()

        # Create instance of MongoPersistence with mocked dependencies
        self.persistence = MongoPersistence()

    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
        self.mongo_patcher.stop()

    def test_init_with_env_vars(self):
        """Test initialization with environment variables"""
        self.mongo_mock.assert_called_once_with(
            host="test-host", port=27017, username="test-user", password="test-password"
        )
        self.assertEqual(self.persistence.collection_name, "hivemind_workflow_states")

    def test_create_workflow_state(self):
        """Test creating a new workflow state"""
        # Mock the insert_one result
        mock_result = MagicMock()
        mock_result.inserted_id = ObjectId("507f1f77bcf86cd799439011")
        self.collection_mock.insert_one.return_value = mock_result

        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community",
            query="test query",
            chat_id="test-chat",
            enable_answer_skipping=True,
        )

        self.assertEqual(workflow_id, "507f1f77bcf86cd799439011")
        self.collection_mock.insert_one.assert_called_once()
        
        # Check the inserted document structure
        inserted_doc = self.collection_mock.insert_one.call_args[0][0]
        self.assertEqual(inserted_doc["communityId"], "test-community")
        self.assertEqual(inserted_doc["question"]["message"], "test query")
        self.assertEqual(inserted_doc["chatId"], "test-chat")
        self.assertTrue(inserted_doc["enableAnswerSkipping"])
        self.assertEqual(inserted_doc["currentStep"], "initialized")
        self.assertEqual(inserted_doc["status"], "running")
        self.assertIn("route", inserted_doc)
        self.assertIn("response", inserted_doc)
        self.assertIn("metadata", inserted_doc)
        self.assertIn("steps", inserted_doc)

    def test_update_workflow_step(self):
        """Test updating workflow step"""
        # Mock the update_one result
        mock_result = MagicMock()
        mock_result.modified_count = 1
        self.collection_mock.update_one.return_value = mock_result

        success = self.persistence.update_workflow_step(
            workflow_id="507f1f77bcf86cd799439011",
            step_name="test_step",
            step_data={"key": "value"},
            status="running",
        )

        self.assertTrue(success)
        self.collection_mock.update_one.assert_called_once()
        
        # Check the update operation
        call_args = self.collection_mock.update_one.call_args
        self.assertEqual(call_args[0][0], {"_id": ObjectId("507f1f77bcf86cd799439011")})
        
        update_data = call_args[0][1]
        self.assertIn("$push", update_data)
        self.assertIn("$set", update_data)
        self.assertEqual(update_data["$set"]["currentStep"], "test_step")

    def test_update_response(self):
        """Test updating response"""
        # Mock the update_one result
        mock_result = MagicMock()
        mock_result.modified_count = 1
        self.collection_mock.update_one.return_value = mock_result

        success = self.persistence.update_response(
            workflow_id="507f1f77bcf86cd799439011",
            response_message="Test answer",
            status="completed",
        )

        self.assertTrue(success)
        self.collection_mock.update_one.assert_called_once()
        
        # Check the update operation
        call_args = self.collection_mock.update_one.call_args
        self.assertEqual(call_args[0][0], {"_id": ObjectId("507f1f77bcf86cd799439011")})
        
        update_data = call_args[0][1]
        self.assertIn("$set", update_data)
        self.assertEqual(update_data["$set"]["response.message"], "Test answer")
        self.assertEqual(update_data["$set"]["status"], "completed")

    def test_get_workflow_state(self):
        """Test getting workflow state"""
        # Mock the find_one result
        mock_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "communityId": "test-community",
            "question": {"message": "test query"},
            "status": "completed",
        }
        self.collection_mock.find_one.return_value = mock_doc

        result = self.persistence.get_workflow_state("507f1f77bcf86cd799439011")

        self.assertIsNotNone(result)
        self.assertEqual(result["_id"], "507f1f77bcf86cd799439011")
        self.assertEqual(result["communityId"], "test-community")
        self.assertEqual(result["question"]["message"], "test query")
        self.assertEqual(result["status"], "completed")

    def test_get_workflow_state_not_found(self):
        """Test getting workflow state that doesn't exist"""
        self.collection_mock.find_one.return_value = None

        result = self.persistence.get_workflow_state("507f1f77bcf86cd799439011")

        self.assertIsNone(result)

    def test_close(self):
        """Test closing the MongoDB connection"""
        self.persistence.close()
        self.client_mock.close.assert_called_once()
