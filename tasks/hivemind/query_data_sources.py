import asyncio
import os
from uuid import uuid1

import nest_asyncio
from dotenv import load_dotenv
from tc_temporal_backend.client import TemporalClient
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload

nest_asyncio.apply()

from crewai.tools import BaseTool


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


class RAGPipelineTool(BaseTool):
    name: str = "RAG pipeline tool"
    description: str = (
        "This tool implements a Retrieval-Augmented Generation (RAG) pipeline which "
        "queries available data sources to provide accurate answers to user queries. "
    )

    @classmethod
    def setup_tools(cls, community_id: str, enable_answer_skipping: bool):
        """
        Setup the tool with the necessary community identifier and the flag to enable answer skipping.
        """
        cls.community_id = community_id
        cls.enable_answer_skipping = enable_answer_skipping
        return cls

    def _run(self, query: str | dict) -> str | None:
        """
        Execute the RAG pipeline by querying the available data sources.

        Parameters
        ------------
        query : str
            The input query string provided by the user.

        Returns
        ----------
        response : str
            The response obtained after querying the data sources.
        """
        query_data_sources = QueryDataSources(
            community_id=self.community_id,
            enable_answer_skipping=self.enable_answer_skipping,
        )

        # manually handling the edge case of the LLM
        # it sometimes gives dictionary (hallucinating)
        if isinstance(query, str):
            response = asyncio.run(query_data_sources.query(query))
        else:
            response = asyncio.run(query_data_sources.query(query["description"]))
        return response
