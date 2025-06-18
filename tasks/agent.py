import logging
from datetime import timedelta

from temporalio import activity, workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from crewai.crews.crew_output import CrewOutput
    from tasks.hivemind.agent import AgenticHivemindFlow
    from tasks.redis_memory import RedisMemory
    from tasks.mongo_persistence import MongoPersistence
    from tc_temporal_backend.schema.hivemind import HivemindQueryPayload


@activity.defn
async def run_hivemind_agent_activity(
    payload: HivemindQueryPayload,
) -> str | None:
    """
    Activity that instantiates and runs the Crew.ai Flow (AgenticHivemindFlow).
    It places the resulting answer into payload.content.response.
    """

    # Initialize MongoDB persistence
    mongo_persistence = MongoPersistence()
    workflow_id = None
    
    try:
        # Create initial workflow state in MongoDB
        workflow_id = mongo_persistence.create_workflow_state(
            community_id=payload.community_id,
            query=payload.query,
            chat_id=getattr(payload, 'chat_id', None),
            enable_answer_skipping=payload.enable_answer_skipping,
        )
        
        logging.info(f"Created workflow state with ID: {workflow_id}")
        
        # Update step: Initialization
        mongo_persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="initialization",
            step_data={
                "communityId": payload.community_id,
                "query": payload.query,
                "chatId": getattr(payload, 'chat_id', None),
                "enableAnswerSkipping": payload.enable_answer_skipping,
            }
        )

        memory: RedisMemory | None
        chat_history: str | None

        if hasattr(payload, 'chat_id') and payload.chat_id:
            memory = RedisMemory(key=f"conversation:{payload.chat_id}")
            chat_history = memory.get_text()
            
            # Update step: Chat history retrieval
            mongo_persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name="chat_history_retrieval",
                step_data={
                    "chatId": payload.chat_id,
                    "chatHistoryLength": len(chat_history) if chat_history else 0,
                }
            )
        else:
            chat_history = None
            memory = None
            
            # Update step: No chat history
            mongo_persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name="no_chat_history",
                step_data={"reason": "No chat_id provided"}
            )

        # Update step: Flow initialization
        mongo_persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="flow_initialization",
            step_data={
                "flowType": "AgenticHivemindFlow",
                "enableAnswerSkipping": payload.enable_answer_skipping,
            }
        )

        # Instantiate the flow with the user query
        flow = AgenticHivemindFlow(
            community_id=payload.community_id,
            user_query=payload.query,
            enable_answer_skipping=payload.enable_answer_skipping,
            chat_history=chat_history,
            workflow_id=workflow_id,
            mongo_persistence=mongo_persistence,
        )

        # Update step: Flow execution start
        mongo_persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="flow_execution_start",
            step_data={"userQuery": payload.query}
        )

        # Run the flow
        crew_output = await flow.kickoff_async(inputs={"query": payload.query})

        # Update step: Flow execution complete
        mongo_persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="flow_execution_complete",
            step_data={
                "crewOutputType": type(crew_output).__name__,
                "hasOutput": crew_output is not None,
            }
        )

        if isinstance(crew_output, CrewOutput):
            final_answer = crew_output.raw
        elif not payload.enable_answer_skipping:
            final_answer = "No answer was generated."
        else:
            final_answer = None

        # Update step: Answer processing
        mongo_persistence.update_workflow_step(
            workflow_id=workflow_id,
            step_name="answer_processing",
            step_data={
                "answerType": type(final_answer).__name__,
                "answerLength": len(final_answer) if isinstance(final_answer, str) else 0,
                "enableAnswerSkipping": payload.enable_answer_skipping,
            }
        )

        if isinstance(final_answer, str) and "encountered an error" in final_answer.lower():
            logging.error(f"final_answer: {final_answer}")
            fallback_answer = "Looks like things didn't go through. Please give it another go."
            
            # Update step: Error handling
            mongo_persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name="error_handling",
                step_data={
                    "errorType": "crewai_error",
                    "originalAnswer": final_answer,
                    "fallbackAnswer": fallback_answer,
                }
            )
            final_answer = fallback_answer

        if memory and final_answer != "NONE":
            chat = f"User: {payload.query}\nAgent: {final_answer}"
            memory.append_text(chat)
            
            # Update step: Memory update
            mongo_persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name="memory_update",
                step_data={
                    "memoryKey": f"conversation:{payload.chat_id}",
                    "chatEntryLength": len(chat),
                }
            )

        # Update final answer in MongoDB
        if final_answer and final_answer != "NONE":
            mongo_persistence.update_response(
                workflow_id=workflow_id,
                response_message=final_answer,
                status="completed"
            )
        else:
            mongo_persistence.update_response(
                workflow_id=workflow_id,
                response_message="No answer generated",
                status="completed_no_answer"
            )

        if final_answer == "NONE":
            return None
        else:
            return final_answer

    except Exception as e:
        logging.error(f"Error in run_hivemind_agent_activity: {e}")
        
        # Update step: Error occurred
        if workflow_id:
            mongo_persistence.update_workflow_step(
                workflow_id=workflow_id,
                step_name="error_occurred",
                step_data={
                    "errorType": type(e).__name__,
                    "errorMessage": str(e),
                },
                status="failed"
            )
            
            # Update final status
            mongo_persistence.update_response(
                workflow_id=workflow_id,
                response_message=None,
                status="failed"
            )
        
        raise


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
