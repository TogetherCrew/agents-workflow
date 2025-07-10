import logging
from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow, listen, start, router
from crewai.llm import LLM
from tasks.hivemind.classify_question import ClassifyQuestion
from tasks.hivemind.query_data_sources import make_rag_tool
from pydantic import BaseModel
from crewai.tools import tool
from openai import OpenAI
from typing import Optional
from tasks.mongo_persistence import MongoPersistence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent


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
        workflow_id: Optional[str] = None,
        mongo_persistence: MongoPersistence | None = None,
        persistence=None,
        max_retry_count: int = 3,
        **kwargs,
    ) -> None:
        self.enable_answer_skipping = enable_answer_skipping
        self.community_id = community_id
        self.workflow_id = workflow_id
        self.mongo_persistence = mongo_persistence
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
        if not self.enable_answer_skipping:
            self.state.state = "continue"
            return
        
        checker = ClassifyQuestion(enable_reasoning=True)

        # classify using a local model
        question = checker.classify_message(message=self.state.user_query)
        # Persist the local model classification result
        if self.mongo_persistence and self.workflow_id:
            self.mongo_persistence.update_workflow_step(
                workflow_id=self.workflow_id,
                step_name="local_model_classification",
                step_data={
                    "result": question,
                    "model": "local_transformer",
                    "query": self.state.user_query,
                }
            )
        
        if not question:
            self.state.state = "stop"
            return

        # classify using a language model
        is_question = checker.classify_question_lm(message=self.state.user_query)
        # Persist the is_question result and reasoning
        if self.mongo_persistence and self.workflow_id:
            self.mongo_persistence.update_workflow_step(
                workflow_id=self.workflow_id,
                step_name="question_classification",
                step_data={
                    "result": is_question.result,
                    "reasoning": is_question.reasoning,
                    "model": "language_model",
                    "query": self.state.user_query,
                }
            )
        
        if not is_question.result:
            self.state.state = "stop"
            return

        # classify if its a RAG question
        rag_question = checker.classify_message_lm(message=self.state.user_query)
        # Persist the rag_question result and reasoning and score
        if self.mongo_persistence and self.workflow_id:
            self.mongo_persistence.update_workflow_step(
                workflow_id=self.workflow_id,
                step_name="rag_classification",
                step_data={
                    "result": rag_question.result,
                    "score": rag_question.score,
                    "reasoning": rag_question.reasoning,
                    "model": "language_model",
                    "query": self.state.user_query,
                }
            )
        
        self.state.state = "continue" if rag_question.result else "stop"

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
            # Persist the history query classification result
            if self.mongo_persistence and self.workflow_id:
                self.mongo_persistence.update_workflow_step(
                    workflow_id=self.workflow_id,
                    step_name="history_query_classification",
                    step_data={
                        "result": is_history_query,
                        "model": "openai_gpt4",
                        "query": self.state.user_query,
                        "hasChatHistory": True,
                    }
                )

        if is_history_query:
            logging.info("History query detected")
            return "history"
        else:
            logging.info("RAG query detected")
            return "rag"

    @router("rag")
    def do_rag_query(self) -> str:
        llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        rag_tool = make_rag_tool(self.enable_answer_skipping, self.community_id, self.workflow_id)
        tools = [rag_tool]

        SYSTEM_INSTRUCTIONS = """\
        You are a helpful assistant.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_INSTRUCTIONS),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_openai_functions_agent(llm, tools, prompt)

        # Run the agent
        agent_executor = AgentExecutor(
            agent=agent, tools=tools, verbose=True, return_intermediate_steps=False
        )

        result = agent_executor.invoke({"input": self.state.user_query})
        self.state.last_answer = result["output"]
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
