from datetime import timedelta

from temporalio import workflow, activity

from tc_temporal_backend.schema.hivemind import HivemindQueryPayload

from tasks.hivemind.agent import AgenticHivemindFlow


@activity.defn
def run_hivemind_agent_activity(payload: HivemindQueryPayload) -> HivemindQueryPayload:
    """
    Activity that instantiates and runs the Crew.ai Flow (AgenticHivemindFlow).
    It places the resulting answer into payload.content.response.
    """
    # Instantiate the flow with the user query
    # (enable_answer_skipping=False here, but you can make it dynamic)
    flow = AgenticHivemindFlow(
        community_id=payload.community_id,
        user_query=payload.query,
        enable_answer_skipping=payload.enable_answer_skipping,
    )

    # Run the flow
    crew_output = flow.kickoff(inputs={"query": payload.query})

    # crew_output could be None if skipping or any other logic, so handle gracefully
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
    async def run(self, payload: HivemindQueryPayload) -> HivemindQueryPayload:
        # Execute the activity with a timeout
        updated_payload = await workflow.execute_activity(
            run_hivemind_agent_activity,
            payload,
            schedule_to_close_timeout=timedelta(minutes=5)
        )
        return updated_payload
