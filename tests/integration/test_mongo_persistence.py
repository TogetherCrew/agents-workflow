import unittest
import uuid
from dotenv import load_dotenv
from bson import ObjectId
from tasks.mongo_persistence import MongoPersistence

class TestMongoPersistenceIntegration(unittest.TestCase):
    """Integration test cases for the MongoPersistence class that work with real MongoDB"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        load_dotenv()
        
        # Use a test-specific collection name to avoid interfering with production data
        cls.test_collection_name = f"test_internal_messages_{uuid.uuid4().hex[:8]}"
        cls.persistence = MongoPersistence(collection_name=cls.test_collection_name)
        
        # Verify MongoDB connection
        try:
            # Test the connection by trying to access the collection
            cls.persistence.collection.find_one()
            print(f"✅ MongoDB connection successful. Using test collection: {cls.test_collection_name}")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            print("Make sure MongoDB is running and environment variables are set correctly")
            raise

    def setUp(self):
        """Set up each test case"""
        # Clear the test collection before each test
        self.persistence.collection.delete_many({})

    def tearDown(self):
        """Clean up after each test"""
        # Clear the test collection after each test
        self.persistence.collection.delete_many({})

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests"""
        # Drop the test collection
        try:
            cls.persistence.collection.drop()
            print(f"✅ Test collection {cls.test_collection_name} dropped successfully")
        except Exception as e:
            print(f"⚠️ Warning: Could not drop test collection: {e}")

    def test_create_workflow_state(self):
        """Test creating a new workflow state with real MongoDB"""
        # Create a workflow state
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community-123",
            query="What is the weather like today?",
            chat_id="test-chat-456",
            enable_answer_skipping=True,
        )

        # Verify the workflow ID is returned
        self.assertIsNotNone(workflow_id)
        self.assertIsInstance(workflow_id, str)
        
        # Verify the document exists in MongoDB
        doc = self.persistence.collection.find_one({"_id": ObjectId(workflow_id)})
        self.assertIsNotNone(doc)
        
        # Verify the document structure
        self.assertEqual(doc["communityId"], "test-community-123")
        self.assertEqual(doc["question"]["message"], "What is the weather like today?")
        self.assertEqual(doc["chatId"], "test-chat-456")
        self.assertTrue(doc["enableAnswerSkipping"])
        self.assertEqual(doc["currentStep"], "initialized")
        self.assertEqual(doc["status"], "running")
        self.assertIn("route", doc)
        self.assertIn("response", doc)
        self.assertIn("metadata", doc)
        self.assertIn("steps", doc)
        self.assertIn("createdAt", doc)
        self.assertIn("updatedAt", doc)

    def test_create_workflow_state_with_optional_params(self):
        """Test creating workflow state with all optional parameters"""
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community-full",
            query="How do I configure the system?",
            source="slack",
            destination={"queue": "SLACK_HIVEMIND_ADAPTER", "event": "MESSAGE_RECEIVED"},
            filters={"category": "general", "priority": "high"},
            metadata={"user_id": "123", "channel": "general", "timestamp": "2024-01-01"},
            chat_id="test-chat-full",
            enable_answer_skipping=False,
        )

        # Verify the document exists and has correct structure
        doc = self.persistence.collection.find_one({"_id": ObjectId(workflow_id)})
        self.assertIsNotNone(doc)
        
        # Check optional parameters
        self.assertEqual(doc["route"]["source"], "slack")
        self.assertEqual(doc["route"]["destination"]["queue"], "SLACK_HIVEMIND_ADAPTER")
        self.assertEqual(doc["route"]["destination"]["event"], "MESSAGE_RECEIVED")
        self.assertEqual(doc["question"]["filters"]["category"], "general")
        self.assertEqual(doc["question"]["filters"]["priority"], "high")
        self.assertEqual(doc["metadata"]["user_id"], "123")
        self.assertEqual(doc["metadata"]["channel"], "general")
        self.assertEqual(doc["metadata"]["timestamp"], "2024-01-01")
        self.assertFalse(doc["enableAnswerSkipping"])

    def test_update_workflow_step(self):
        """Test updating workflow step with real MongoDB"""
        # First create a workflow state
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community",
            query="Test query",
        )

        # Update with a step
        step_data = {
            "model": "gpt-4",
            "confidence": 0.95,
            "reasoning": "This is a test step"
        }
        
        success = self.persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="test_classification",
            step_data=step_data,
            status="processing"
        )

        self.assertTrue(success)
        
        # Verify the update in MongoDB
        doc = self.persistence.collection.find_one({"_id": ObjectId(workflow_id)})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["currentStep"], "test_classification")
        self.assertEqual(doc["status"], "processing")
        self.assertEqual(len(doc["steps"]), 1)
        
        # Check the step data
        step = doc["steps"][0]
        self.assertEqual(step["stepName"], "test_classification")
        self.assertEqual(step["data"]["model"], "gpt-4")
        self.assertEqual(step["data"]["confidence"], 0.95)
        self.assertEqual(step["data"]["reasoning"], "This is a test step")
        self.assertIn("timestamp", step)

    def test_update_workflow_step_multiple_steps(self):
        """Test updating workflow with multiple steps"""
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community",
            query="Test query",
        )

        # Add multiple steps
        steps = [
            ("initialization", {"status": "started"}),
            ("classification", {"model": "local", "result": True}),
            ("rag_query", {"sources": ["doc1", "doc2"]}),
            ("response_generation", {"model": "gpt-4", "tokens": 150}),
        ]

        for step_name, step_data in steps:
            success = self.persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name=step_name,
                step_data=step_data,
            )
            self.assertTrue(success)

        # Verify all steps are stored
        doc = self.persistence.collection.find_one({"_id": ObjectId(workflow_id)})
        self.assertEqual(len(doc["steps"]), 4)
        self.assertEqual(doc["currentStep"], "response_generation")
        
        # Check each step
        step_names = [step["stepName"] for step in doc["steps"]]
        self.assertEqual(step_names, ["initialization", "classification", "rag_query", "response_generation"])

    def test_update_response(self):
        """Test updating response with real MongoDB"""
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community",
            query="What is the answer?",
        )

        # Update the response
        response_message = "The answer is 42. This is a comprehensive response that addresses the user's question."
        success = self.persistence.update_response(
            workflow_id=workflow_id,
            response_message=response_message,
            status="completed"
        )

        self.assertTrue(success)
        
        # Verify the response in MongoDB
        doc = self.persistence.collection.find_one({"_id": ObjectId(workflow_id)})
        self.assertIsNotNone(doc)
        self.assertEqual(doc["response"]["message"], response_message)
        self.assertEqual(doc["status"], "completed")

    def test_get_workflow_state(self):
        """Test getting workflow state with real MongoDB"""
        # Create a workflow state
        original_workflow_id = self.persistence.create_workflow_state(
            community_id="test-community",
            query="Test query for retrieval",
            chat_id="test-chat",
            enable_answer_skipping=True,
        )

        # Add some steps
        self.persistence.update_workflow_step(
            workflow_id=original_workflow_id,
            step_name="test_step",
            step_data={"key": "value"}
        )

        # Retrieve the workflow state
        retrieved_doc = self.persistence.get_workflow_state(original_workflow_id)

        # Verify the retrieved document
        self.assertIsNotNone(retrieved_doc)
        self.assertEqual(retrieved_doc["_id"], original_workflow_id)
        self.assertEqual(retrieved_doc["communityId"], "test-community")
        self.assertEqual(retrieved_doc["question"]["message"], "Test query for retrieval")
        self.assertEqual(retrieved_doc["chatId"], "test-chat")
        self.assertTrue(retrieved_doc["enableAnswerSkipping"])
        self.assertEqual(len(retrieved_doc["steps"]), 1)
        self.assertEqual(retrieved_doc["steps"][0]["stepName"], "test_step")

    def test_get_workflow_state_not_found(self):
        """Test getting workflow state that doesn't exist"""
        # Try to get a non-existent workflow
        fake_id = "507f1f77bcf86cd799439011"  # Valid ObjectId format but doesn't exist
        result = self.persistence.get_workflow_state(fake_id)
        self.assertIsNone(result)

    def test_complete_workflow_lifecycle(self):
        """Test a complete workflow lifecycle from creation to completion"""
        # 1. Create workflow state
        workflow_id = self.persistence.create_workflow_state(
            community_id="test-community-lifecycle",
            query="How do I deploy the application?",
            source="discord",
            destination={"queue": "DISCORD_ADAPTER", "event": "QUESTION_RECEIVED"},
            metadata={"user": "testuser", "channel": "deployment"},
            chat_id="lifecycle-chat",
            enable_answer_skipping=False,
        )

        # 2. Add classification step
        self.persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="question_classification",
            step_data={
                "model": "local_transformer",
                "result": True,
                "confidence": 0.92,
                "reasoning": "This is a deployment-related question"
            }
        )

        # 3. Add RAG query step
        self.persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="rag_query",
            step_data={
                "sources": ["deployment_guide.md", "troubleshooting.md"],
                "query": "How do I deploy the application?",
                "results_count": 5
            }
        )

        # 4. Add response generation step
        self.persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="response_generation",
            step_data={
                "model": "gpt-4",
                "tokens_used": 245,
                "generation_time": 2.3
            }
        )

        # 5. Update with final response
        final_response = "To deploy the application, follow these steps: 1. Build the project, 2. Run tests, 3. Deploy to staging, 4. Deploy to production."
        self.persistence.update_response(
            workflow_id=workflow_id,
            response_message=final_response,
            status="completed"
        )

        # 6. Verify the complete workflow state
        final_doc = self.persistence.get_workflow_state(workflow_id)
        self.assertIsNotNone(final_doc)
        self.assertEqual(final_doc["status"], "completed")
        self.assertEqual(final_doc["response"]["message"], final_response)
        self.assertEqual(len(final_doc["steps"]), 3)
        self.assertEqual(final_doc["currentStep"], "response_generation")
        
        # Verify all the data is preserved
        self.assertEqual(final_doc["communityId"], "test-community-lifecycle")
        self.assertEqual(final_doc["route"]["source"], "discord")
        self.assertEqual(final_doc["metadata"]["user"], "testuser")
        self.assertFalse(final_doc["enableAnswerSkipping"])

    def test_error_handling_invalid_object_id(self):
        """Test error handling with invalid ObjectId"""
        # Test with invalid ObjectId format
        invalid_id = "invalid-id-format"
        
        # These should handle the error gracefully
        result = self.persistence.get_workflow_state(invalid_id)
        self.assertIsNone(result)
        
        # Update operations should return False for invalid IDs
        success = self.persistence.update_workflow_step(
            workflow_id=invalid_id,
            step_name="test",
            step_data={}
        )
        self.assertFalse(success)
        
        success = self.persistence.update_response(
            workflow_id=invalid_id,
            response_message="test"
        )
        self.assertFalse(success)
