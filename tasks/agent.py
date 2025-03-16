import logging
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from crewai.crews.crew_output import CrewOutput
    from tasks.hivemind.agent import AgenticHivemindFlow
    from tc_temporal_backend.schema.hivemind import HivemindQueryPayload


@activity.defn
async def run_hivemind_agent_activity(
    payload: HivemindQueryPayload,
) -> str | None:
    """
    Activity that instantiates and runs the Crew.ai Flow (AgenticHivemindFlow).
    It places the resulting answer into payload.content.response.
    """
    # Instantiate the flow with the user query
    flow = AgenticHivemindFlow(
        community_id=payload.community_id,
        user_query=payload.query,
        enable_answer_skipping=payload.enable_answer_skipping,
    )

    # Run the flow
    crew_output = await flow.kickoff_async(inputs={"query": payload.query})

    if isinstance(crew_output, CrewOutput):
        final_answer = crew_output.raw
    elif not payload.enable_answer_skipping:
        final_answer = "No answer was generated."
    else:
        final_answer = None

    if isinstance(final_answer, str) and "encountered an error" in final_answer.lower():
        logging.error(f"final_answer: {final_answer}")
        final_answer = "Looks like things didn't go through. Please give it another go."

    if final_answer == "NONE":
        return None
    else:
        return final_answer


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
