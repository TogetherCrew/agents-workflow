import os
from uuid import uuid1

from crewai.tools import tool
from dotenv import load_dotenv
from tc_temporal_backend.client import TemporalClient
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload


class QueryDataSources:
    def __init__(self, community_id: str, enable_answer_skipping: bool):
        self.community_id = community_id
        self.enable_answer_skipping = enable_answer_skipping

    @tool
    async def query(self, query: str) -> str:
        """
        query data sources for the given community
        """
        client = await TemporalClient().get_client()

        payload = HivemindQueryPayload(
            community_id=self.community_id,
            query=query,
            enable_answer_skipping=self.enable_answer_skipping,
        )

        hivemind_queue = self.load_hivemind_queue()
        result = await client.execute_workflow(
            "HivemindWorkflow",
            payload,
            id=f"hivemind-query-{self.community_id}-{uuid1()}",
            task_queue=hivemind_queue,
        )

        return result

    def load_hivemind_queue(self) -> str:
        """
        load the hivemind queue name
        """
        load_dotenv()
        hivemind_queue = os.getenv("TEMPORAL_HIVEMIND_TASK_QUEUE")
        if not hivemind_queue:
            raise ValueError("env `TEMPORAL_HIVEMIND_TASK_QUEUE` is not loaded!")

        return hivemind_queue
