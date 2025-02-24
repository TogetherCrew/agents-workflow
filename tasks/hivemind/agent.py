from crewai.flow.flow import Flow, listen, start
from crewai.llm import LLM
from crewai.crews.crew_output import CrewOutput
from crewai import Agent, Task, Crew
from tasks.hivemind.check_question import CheckQuestion
from tasks.hivemind.query_data_sources import QueryDataSources


class AgenticHivemindFlow(Flow):
    model = "gpt-4o-mini"

    def __init__(
            self,
            user_query: str,
            community_id: str,
            enable_answer_skipping: bool = False,
            persistence = None,
            **kwargs
        ) -> None:
        self.user_query = user_query
        self.enable_answer_skipping = enable_answer_skipping
        self.community_id = community_id
        super().__init__(persistence, **kwargs)

    @start()
    def detect_question(self) -> str:
        if self.enable_answer_skipping:
            checker = CheckQuestion(model=self.model)
            rag_question = checker.check_question(message=self.user_query)
            if rag_question:
                return self.user_query
            else:
                return None
        else:
            return self.user_query

    @listen(detect_question)
    def query(self, query: str | None) -> CrewOutput:
        if query:
            query_data_sources = QueryDataSources(
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
                tools=[query_data_sources.query],
                allow_delegation=True,
                llm=LLM(model="gpt-4o-mini")
            )

            math_task = Task(
                description=(
                    "Answer the following query. If the query is specific to community data, use the tool to retrieve updated information; "
                    "otherwise, answer using your internal knowledge. Query: {query}"
                ),
                expected_output="The answer of the given query",
                agent=q_a_bot_agent  
            )

            crew = Crew(
                agents=[q_a_bot_agent],
                tasks=[math_task],
                verbose=True
            )

            crew_output = crew.kickoff(inputs={"query": query})

            return crew_output
        else:
            return None