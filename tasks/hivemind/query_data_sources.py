import asyncio
import os
from uuid import uuid1

import nest_asyncio
from dotenv import load_dotenv
from tc_temporal_backend.client import TemporalClient
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload

nest_asyncio.apply()

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class QueryDataSources:
    def __init__(self, community_id: str, enable_answer_skipping: bool):
        self.community_id = community_id
        self.enable_answer_skipping = enable_answer_skipping

    async def query(self, query: str) -> str:
        """
        query data sources for the given community

        Parameters
        ------------
        query : str
            the query to search for
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


class QueryDataSourcesInput(BaseModel):
    """Input schema for QueryDataSourcesInput."""

    query: str = Field(..., description="The query to apply on data sources.")


class QueryDataSourcesTool(BaseTool):
    name: str = "Query Data sources"
    description: str = "Query available data sources to find the answer."
    args_schema: Type[BaseModel] = QueryDataSourcesInput

    @classmethod
    def setup_tools(cls, community_id: str, enable_answer_skipping: bool):
        cls.community_id = community_id
        cls.enable_answer_skipping = enable_answer_skipping

        return cls

    def _run(self, query: str) -> str:
        query_data_sources = QueryDataSources(
            community_id=self.community_id,
            enable_answer_skipping=self.enable_answer_skipping,
        )
        response = asyncio.run(query_data_sources.query(query))
        return response
