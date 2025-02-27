from datetime import timedelta

from temporalio import activity, workflow

with workflow.unsafe.imports_passed_through():
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

    if crew_output:
        final_answer = crew_output
    elif not payload.enable_answer_skipping:
        final_answer = "No answer was generated."
    else:
        final_answer = None

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
            schedule_to_close_timeout=timedelta(minutes=5),
        )
        return updated_payload
