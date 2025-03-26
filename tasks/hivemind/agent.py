from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput
from crewai.flow.flow import Flow, listen, start, router
from crewai.llm import LLM
from tasks.hivemind.classify_question import ClassifyQuestion
from tasks.hivemind.query_data_sources import RAGPipelineTool
from tasks.hivemind.answer_validator import AnswerValidator
from pydantic import BaseModel


class AgenticFlowState(BaseModel):
    user_query: str = ""
    retry_count: int = 0
    last_answer: CrewOutput | None = None
    state: str = "continue"


class AgenticHivemindFlow(Flow[AgenticFlowState]):
    model = "gpt-4o"

    def __init__(
        self,
        user_query: str,
        community_id: str,
        enable_answer_skipping: bool = False,
        persistence=None,
        max_retry_count: int = 3,
        **kwargs,
    ) -> None:
        self.enable_answer_skipping = enable_answer_skipping
        self.community_id = community_id
        self.max_retry_count = max_retry_count
        super().__init__(persistence, **kwargs)

        self.state.user_query = user_query

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

    @listen("continue")
    def query(self) -> CrewOutput | str:
        query_data_source_tool = RAGPipelineTool.setup_tools(
            community_id=self.community_id,
            enable_answer_skipping=self.enable_answer_skipping,
        )

        q_a_bot_agent = Agent(
            role="Q&A Bot",
            goal=(
                "You decide when to rely on your internal knowledge and when to retrieve real-time data. "
                "For queries that are not specific to community data, answer using your own LLM knowledge."
            ),
            backstory=(
                "You are an intelligent agent capable of answering questions using either your internal LLM knowledge "
                "or a Retrieval-Augmented Generation (RAG) pipeline to fetch community-specific data."
            ),
            tools=[
                query_data_source_tool(result_as_answer=True),
            ],
            allow_delegation=True,
            llm=LLM(model="gpt-4o-mini"),
        )

        rag_task = Task(
            description=(
                "Answer the following query. If the query is specific to community data, use the tool to retrieve updated information; "
                f"otherwise, answer using your internal knowledge. Query: {self.state.user_query}"
            ),
            expected_output="The answer of the given query",
            agent=q_a_bot_agent,
        )

        crew = Crew(agents=[q_a_bot_agent], tasks=[rag_task], verbose=True)
        crew_output = crew.kickoff(inputs={"query": self.state.user_query})

        # Store the latest crew output and increment retry count
        self.state.last_answer = crew_output
        self.state.retry_count += 1

        return crew_output

    @router(query)
    def check_answer_validity(self, crew_output: CrewOutput) -> str:
        if crew_output.raw == "NONE":
            return "stop"
        else:
            checker = AnswerValidator()
            validity = checker.check_answer_validity(
                question=self.state.user_query, answer=crew_output.raw
            )
            if validity is False:
                return "refine"
            else:
                return "stop"

    @router("refine")
    def refine_question(self) -> CrewOutput | None:
        """
        - If the answer is invalid and we've tried fewer than max_retry_count times,
          refine the question and re-query.
        - If the answer is invalid and we've reached or exceeded max_retry_count,
          return the last answer we have.
        """

        # Check if we've reached the max retry limit
        if self.state.retry_count < self.max_retry_count:
            refine_agent = Agent(
                role="Question Refiner",
                goal="Analyze and refine unclear questions to get better answers",
                backstory=(
                    "You are an expert at understanding and reformulating questions "
                    "to make them more specific and answerable."
                ),
                llm=LLM(model=self.model),
            )

            refine_task = Task(
                description=(
                    f"The original question '{self.state.user_query}' did not yield a satisfactory answer. "
                    "Please analyze the question and reformulate it to be more specific and clear. "
                    "Consider breaking down complex questions or adding context if needed."
                ),
                expected_output="A refined, more specific version of the original question",
                agent=refine_agent,
            )

            crew = Crew(agents=[refine_agent], tasks=[refine_task], verbose=True)

            refined_output = crew.kickoff()

            self.state.user_query = refined_output.raw
            return "continue"
        else:
            return "stop"
