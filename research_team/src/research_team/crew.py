from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool

# Import constants
from .constants import RESEARCHER_MODEL, REPORTING_ANALYST_MODEL

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class ResearchTeam():
    """ResearchTeam crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        super().__init__()
        # Create different LLM instances for different agents
        self.researcher_llm = LLM(model=RESEARCHER_MODEL)
        self.reporting_analyst_llm = LLM(model=REPORTING_ANALYST_MODEL)

    # Learn more about YAML configuration files here:
    # Agents:
    # https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks:
    # https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents,
    # you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=False,
            tools=[SerperDevTool()],
            llm=self.researcher_llm,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=False,
            llm=self.reporting_analyst_llm,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchTeam crew"""
        # To learn how to add knowledge sources to your crew,
        # check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            # Automatically created by the @agent decorator
            agents=self.agents,
            # Automatically created by the @task decorator
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical,
            # In case you wanna use that instead
            # https://docs.crewai.com/how-to/Hierarchical/
        )
