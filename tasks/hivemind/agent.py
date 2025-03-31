from crewai import Agent, Crew, Task
from crewai.crews.crew_output import CrewOutput, TaskOutput
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
        # Setup RAG pipeline tool
        query_data_source_tool = RAGPipelineTool.setup_tools(
            community_id=self.community_id,
            enable_answer_skipping=self.enable_answer_skipping,
        )

        # Q&A Bot Agent setup
        q_a_bot_agent = Agent(
            role="RAG Bot",
            goal=(
                "Retrieve and provide relevant information from the community data. "
                "Your final response must not exceed 250 words."
            ),
            backstory=(
                "You are an intelligent agent specialized in retrieving and presenting community-specific data "
                "using a Retrieval-Augmented Generation (RAG) pipeline."
            ),
            tools=[query_data_source_tool(result_as_answer=True)],
            allow_delegation=True,
            llm=LLM(model="gpt-4o-mini"),
        )

        # Direct LLM Agent setup
        llm_agent = Agent(
            role="Direct LLM",
            goal="Provide accurate answers using internal knowledge only",
            backstory="You are an intelligent agent that answers questions using your built-in knowledge",
            llm=LLM(model="gpt-4o-mini"),
        )

        # Create tasks for both agents
        rag_task = Task(
            description=(
                "Answer the following query using a maximum of 250 words. "
                "Use the provided tool to retrieve community-specific information.\n\n"
                f"Query: {self.state.user_query}"
            ),
            expected_output="The answer to the given query based on community data, not exceeding 250 words",
            agent=q_a_bot_agent,
        )

        llm_task = Task(
            description=(
                "Answer the following query using a maximum of 250 words. "
                f"Answer the following query using your internal knowledge only: {self.state.user_query}"
            ),
            expected_output="A clear and concise answer to the query, not exceeding 250 words",
            agent=llm_agent,
        )

        # Create and run crews in parallel
        crew = Crew(
            agents=[q_a_bot_agent, llm_agent],
            tasks=[rag_task, llm_task],
            verbose=True,
            process_parallel=True,
        )

        crew_outputs = crew.kickoff(inputs={"query": self.state.user_query})

        # Compare answers and select the best one
        final_output = self._compare_answers(
            rag_answer=crew_outputs.tasks_output[0],
            llm_answer=crew_outputs.tasks_output[1],
            question=self.state.user_query,
        )

        # Store the final output and increment retry count
        self.state.last_answer = final_output
        self.state.retry_count += 1

        return final_output

    def _compare_answers(
        self, rag_answer: TaskOutput, llm_answer: TaskOutput, question: str
    ) -> TaskOutput:
        compare_agent = Agent(
            role="Answer Comparator",
            goal="Compare answers and select the most informative one",
            backstory="You are an expert at evaluating answer quality and relevance",
            llm=LLM(model=self.model),
        )

        compare_task = Task(
            description=(
                f"Compare these two answers to the question: '{question}'\n\n"
                f"RAG Answer: {rag_answer.raw}\n\n"
                f"LLM Answer: {llm_answer.raw}\n\n"
                "Determine if the RAG answer is not informative at all and send back the LLM Answer."
                "Consider factors like specificity, accuracy, and completeness."
            ),
            expected_output="The better answer between the two options",
            agent=compare_agent,
        )

        crew = Crew(agents=[compare_agent], tasks=[compare_task], verbose=True)
        comparison_result = crew.kickoff()

        # Return the appropriate answer based on the comparison
        if "RAG" in comparison_result.raw.upper():
            return rag_answer
        else:
            return llm_answer

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
