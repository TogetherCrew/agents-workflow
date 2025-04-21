from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from tc_temporal_backend.schema.hivemind import HivemindQueryPayload
    from tasks.hivemind.activities import run_hivemind_agent_activity

@workflow.defn
class AgenticHivemindTemporalWorkflow:
    """
    A Temporal workflow that accepts a Payload, calls the run_hivemind_agent_activity activity,
    and returns the updated Payload (containing the Crew.ai answer).
    """

    @workflow.run
    async def run(self, payload: HivemindQueryPayload) -> str | None:
        # Execute the activity with a timeout
        updated_payload = await workflow.execute_activity(
            run_hivemind_agent_activity,
            payload,
            schedule_to_close_timeout=timedelta(minutes=6),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        return updated_payload
