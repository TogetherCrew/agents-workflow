import asyncio
import os
import logging

import nest_asyncio
from dotenv import load_dotenv
from typing import Optional, Callable
from tc_temporal_backend.client import TemporalClient
from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
from temporalio.common import RetryPolicy
from temporalio.client import WorkflowFailureError

nest_asyncio.apply()



class QueryDataSources:
    def __init__(self, community_id: str, enable_answer_skipping: bool, workflow_id: Optional[str] = None):
        self.community_id = community_id
        self.enable_answer_skipping = enable_answer_skipping
        self.workflow_id = workflow_id

    async def query(self, query: str) -> str | None:
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
            workflow_id=self.workflow_id,
        )

        # Add workflow_id to payload if available
        if self.workflow_id:
            payload.workflow_id = self.workflow_id

        hivemind_queue = self.load_hivemind_queue()
        try:
            result = await client.execute_workflow(
                "HivemindWorkflow",
                payload,
                id=f"hivemind-query-{self.community_id}-{self.workflow_id}",
                task_queue=hivemind_queue,
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
        except WorkflowFailureError as e:
            logging.error(f"WorkflowFailureError: {e} for workflow {self.workflow_id}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Exception: {e} for workflow {self.workflow_id}", exc_info=True)
            return None

        # Normalize Temporal failure-shaped responses that may be returned as data
        if isinstance(result, dict) and (
            "workflowExecutionFailedEventAttributes" in result or "failure" in result
        ):
            logging.error(f"WorkflowFailureError: {result} for workflow {self.workflow_id}", exc_info=True)
            return None
        if isinstance(result, str) and "workflowExecutionFailedEventAttributes" in result:
            logging.error(f"WorkflowFailureError: {result} for workflow {self.workflow_id}", exc_info=True)
            return None

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


def make_rag_tool(enable_answer_skipping: bool, community_id: str, workflow_id: Optional[str] = None) -> Callable:
    """
    Make the RAG pipeline tool.
    Passing the arguments to the tool instead of relying on the LLM to pass them (making the work for LLM easier)

    Args:
        enable_answer_skipping (bool): The flag to enable answer skipping.
        community_id (str): The community ID.
        workflow_id (Optional[str]): The workflow ID.

    Returns:
        Callable: The RAG pipeline tool.
    """
    try:
        from langchain.tools import tool as lc_tool  # type: ignore
    except Exception:
        # Fallback no-op decorator if LangChain is not installed/required
        def lc_tool(*_args, **_kwargs):
            def decorator(func):
                return func
            return decorator
    @lc_tool(return_direct=True)
    def get_rag_answer(query: str) -> str:
        """
        Get the answer from the RAG pipeline

        Args:
            query (str): The input query string provided by the user.

        Returns:
            str: The answer to the query.
        """
        query_data_sources = QueryDataSources(
            community_id=community_id,
            enable_answer_skipping=enable_answer_skipping,
            workflow_id=workflow_id,
        )
        response = asyncio.run(query_data_sources.query(query))

        # crewai doesn't let the tool to return `None`
        if response is None:
            return "NONE"
        else:
            return response

    # returing the tool function
    return get_rag_answer