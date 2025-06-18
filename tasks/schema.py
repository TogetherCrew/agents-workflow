from pydantic import BaseModel, Field


class AgentQueryPayload(BaseModel):
    community_id: str = Field(
        ..., description="the community id data to use for answering"
    )
    query: str = Field(..., description="the user query to ask llm")
    enable_answer_skipping: bool = Field(
        False,
        description=(
            "skip answering questions with non-relevant retrieved information"
            "having this, it could provide `None` for response and source_nodes"
        ),
    )
    chat_id: str = Field(
        default="",
        description="the chat id to use for answering",
    )
