import logging

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pymongo.database import Database
from pymongo.collection import Collection
from bson import ObjectId
from tc_hivemind_backend.db.mongo import MongoSingleton

class MongoPersistence:
    """A class for persisting workflow state data to MongoDB."""

    def __init__(self, database_name: str = "hivemind", collection_name: str = "internal_messages"):
        """Initialize MongoDB connection using environment variables.

        Parameters
        ----------
        collection_name : str
            The MongoDB collection name to use for storing workflow states
        """
        self.collection_name = collection_name
        self.client = MongoSingleton.get_instance().get_client()
        self.db: Database = self.client[database_name]
        self.collection: Collection = self.db[self.collection_name]

    def create_workflow_state(
        self,
        community_id: str,
        query: str,
        source: str = "temporal",
        destination: dict[str, str] | None = None,
        filters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chat_id: Optional[str] = None,
        enable_answer_skipping: bool = False,
    ) -> str:
        """Create a new workflow state document and return its ID.

        Parameters
        ----------
        community_id : str
            The community identifier
        query : str
            The user query
        source : str
            The source of the request (e.g., "discord")
        destination : dict[str, str] | None
            The destination of the request (e.g., {"queue": "DISCORD_HIVEMIND_ADAPTER", "event": "QUESTION_COMMAND_RECEIVED"})
        filters : Optional[Dict[str, Any]]
            Optional filters for the question
        metadata : Optional[Dict[str, Any]]
            Optional metadata from the client side
        chat_id : Optional[str]
            The chat identifier
        enable_answer_skipping : bool
            Whether answer skipping is enabled

        Returns
        -------
        str
            The MongoDB document ID as a string
        """
        try:
            workflow_state = {
                "communityId": community_id,
                "route": {
                    "source": source,
                    "destination": destination,
                },
                "question": {
                    "message": query,
                    "filters": filters
                },
                "response": None,
                "metadata": metadata or {},
                "createdAt": datetime.now(tz=timezone.utc),
                "updatedAt": datetime.now(tz=timezone.utc),
                "steps": [],
                "currentStep": "initialized",
                "status": "running",
                "chatId": chat_id,
                "enableAnswerSkipping": enable_answer_skipping,
            }
            
            result = self.collection.insert_one(workflow_state)
            return str(result.inserted_id)
        except Exception as e:
            logging.error(f"Error creating workflow state: {e}")
            raise

    def update_workflow_step(
        self,
        workflow_id: str,
        step_name: str,
        step_data: Dict[str, Any],
        status: str = "running",
    ) -> bool:
        """Update the workflow state with a new step.

        Parameters
        ----------
        workflow_id : str
            The MongoDB document ID
        step_name : str
            The name of the current step
        step_data : Dict[str, Any]
            The data for this step
        status : str
            The current status of the workflow

        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        try:
            step_entry = {
                "stepName": step_name,
                "timestamp": datetime.now(tz=timezone.utc),
                "data": step_data,
            }

            update_data = {
                "$push": {"steps": step_entry},
                "$set": {
                    "currentStep": step_name,
                    "status": status,
                    "updatedAt": datetime.now(tz=timezone.utc),
                }
            }
            
            result = self.collection.update_one(
                {"_id": ObjectId(workflow_id)}, update_data
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating workflow step: {e}")
            return False

    def update_response(
        self,
        workflow_id: str,
        response_message: str,
        status: str = "completed",
    ) -> bool:
        """Update the workflow state with the response message.

        Parameters
        ----------
        workflow_id : str
            The MongoDB document ID
        response_message : str
            The response message from the workflow
        status : str
            The final status of the workflow

        Returns
        -------
        bool
            True if update was successful, False otherwise
        """
        try:
            update_data = {
                "$set": {
                    "response.message": response_message,
                    "status": status,
                    "updatedAt": datetime.now(tz=timezone.utc),
                }
            }
            
            result = self.collection.update_one(
                {"_id": ObjectId(workflow_id)}, update_data
            )
            return result.modified_count > 0
        except Exception as e:
            logging.error(f"Error updating response: {e}")
            return False

    def get_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the workflow state by ID.

        Parameters
        ----------
        workflow_id : str
            The MongoDB document ID

        Returns
        -------
        Optional[Dict[str, Any]]
            The workflow state document or None if not found
        """
        try:
            document = self.collection.find_one({"_id": ObjectId(workflow_id)})
            if document:
                # Convert ObjectId to string for JSON serialization
                document["_id"] = str(document["_id"])
            return document
        except Exception as e:
            logging.error(f"Error getting workflow state: {e}")
            return None

    def close(self):
        """Close the MongoDB connection."""
        try:
            self.client.close()
        except Exception as e:
            logging.error(f"Error closing MongoDB connection: {e}") 