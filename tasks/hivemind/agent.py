import logging
from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow, listen, start, router
from crewai.llm import LLM
from tasks.hivemind.classify_question import ClassifyQuestion
from tasks.hivemind.query_data_sources import RAGPipelineTool
from crewai.process import Process
from pydantic import BaseModel
from crewai.tools import tool
from openai import OpenAI


class AgenticFlowState(BaseModel):
    user_query: str = ""
    retry_count: int = 0
    last_answer: CrewOutput | None = None
    state: str = "continue"
    chat_history: str | None = None


class AgenticHivemindFlow(Flow[AgenticFlowState]):
    model = "o4-mini-2025-04-16"

    def __init__(
        self,
        user_query: str,
        community_id: str,
        enable_answer_skipping: bool = False,
        chat_history: str | None = None,
        persistence=None,
        max_retry_count: int = 3,
        **kwargs,
    ) -> None:
        self.enable_answer_skipping = enable_answer_skipping
        self.community_id = community_id
        self.max_retry_count = max_retry_count
        super().__init__(persistence, **kwargs)

        self.state.user_query = user_query

        if not chat_history:
            logging.warning(
                "No chat history provided. "
                "All answers will be passed either to RAG or LLM's general knowledge!"
            )

        self.state.chat_history = chat_history

    @start()
    def detect_question(self):
        if self.enable_answer_skipping:
            checker = ClassifyQuestion()
            question = checker.classify_message(message=self.state.user_query)

            if question:
                rag_question = checker.classify_message_lm(
                    message=self.state.user_query
                )
                self.state.state = "continue" if rag_question else "stop"
            else:
                self.state.state = "stop"
        else:
            self.state.state = "continue"

    @router(detect_question)
    def route_start(self) -> str:
        if self.state.state == "continue":
            return "continue"
        elif self.state.state == "stop":
            return "stop"

    @listen("stop")
    def detect_stop_state(self) -> CrewOutput | None:
        return self.state.last_answer

    @router("continue")
    def detect_question_type(self) -> str:
        is_history_query = False
        if self.state.chat_history:
            is_history_query = self.classify_query(self.state.user_query)

        if is_history_query:
            logging.info("History query detected")
            return "history"
        else:
            logging.info("RAG query detected")
            return "rag"

    @router("rag")
    def do_rag_query(self) -> str:
        query_data_source_tool = RAGPipelineTool.setup_tools(
            community_id=self.community_id,
            enable_answer_skipping=self.enable_answer_skipping,
        )

        q_a_bot_agent = Agent(
            role="Q&A Bot",
            goal=(
                "You decide when to rely on your internal knowledge and when to retrieve real-time data. "
                "For queries that are not specific to community data, answer using your own LLM knowledge. "
                "Your final response must not exceed 250 words."
            ),
            backstory=(
                "You are an intelligent agent capable of giving concise answers to questions."
            ),
            allow_delegation=True,
            llm=LLM(model="gpt-4o-mini-2024-07-18"),
        )
        rag_task = Task(
            description=(
                "Answer the following query using a maximum of 250 words. "
                "If the query is specific to community data, use the tool to retrieve updated information; "
                f"otherwise, answer using your internal knowledge.\n\nQuery: {self.state.user_query}"
            ),
            expected_output="A clear, well-structured answer under 250 words that directly addresses the query using appropriate information sources",
            agent=q_a_bot_agent,
            tools=[
                query_data_source_tool(result_as_answer=True),
            ],
        )

        crew = Crew(
            agents=[q_a_bot_agent],
            tasks=[rag_task],
            process=Process.hierarchical,
            manager_llm=LLM(model="gpt-4o-mini-2024-07-18"),
            verbose=True,
        )

        crew_output = crew.kickoff()

        # Store the latest crew output and increment retry count
        self.state.last_answer = crew_output
        self.state.retry_count += 1

        return "stop"

    @router("history")
    def do_history_query(self) -> str:
        q_a_bot_agent = Agent(
            role="History bot",
            goal=(
                "You are an intelligent agent capable of giving concise answers to questions about chat history."
            ),
            backstory=(
                "You are an intelligent agent capable of giving concise answers to questions."
            ),
            llm=LLM(model="gpt-4o-mini-2024-07-18"),
        )

        @tool
        def get_chat_history() -> str:
            "fetch chat history"
            return f"Chat History: {self.state.chat_history}\n"

        history_task = Task(
            description=f"Answer the following query about chat history: {self.state.user_query}",
            expected_output="A response that incorporates the relevant historical context",
            agent=q_a_bot_agent,
            tools=[get_chat_history],
        )

        crew = Crew(
            agents=[q_a_bot_agent],
            tasks=[history_task],
            verbose=True,
        )

        crew_output = crew.kickoff()

        # Store the latest crew output and increment retry count
        self.state.last_answer = crew_output
        self.state.retry_count += 1

        return "stop"

    def classify_query(self, query: str) -> bool:
        """
        Use LLM to determine if the query is about chat history or past conversations.
        """

        class Decision(BaseModel):
            is_history_query: bool

        client = OpenAI()

        system_prompt = (
            "You are an expert at analyzing user queries to determine "
            "if they are about chat history or they require inernal/external knowledge."
        )

        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            response_format=Decision,
            temperature=0,
        )

        decision = completion.choices[0].message.parsed
        return decision.is_history_query
