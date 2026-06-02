# marble/engine/engine.py

"""
The core engine module that coordinates agents within the environment.
"""
import json
from typing import Any, Dict, List, Optional, Union

from marble.agent import BaseAgent
from marble.configs.config import Config
from marble.engine.engine_planner import EnginePlanner
from marble.environments import (
    BaseEnvironment,
    CodingEnvironment,
    DBEnvironment,
    MinecraftEnvironment,
    ResearchEnvironment,
    WebEnvironment,
    WorldSimulationEnvironment,
)
from marble.evaluator.evaluator import Evaluator
from marble.graph.agent_graph import AgentGraph
from marble.memory.base_memory import BaseMemory
from marble.memory.shared_memory import SharedMemory
from marble.utils.logger import get_logger

EnvType = Union[
    BaseEnvironment,
    WebEnvironment,
    ResearchEnvironment,
    WorldSimulationEnvironment,
    MinecraftEnvironment,
    DBEnvironment,
    CodingEnvironment,
]
AgentType = Union[BaseAgent]


class Engine:
    """
    The Engine class orchestrates the simulation, coordinating agents and the environment.
    """

    def _read_code_from_file(self, file_path: str) -> str:
        """
        Read code from a specified file path.

        Args:
            file_path (str): File path

        Returns:
            str: File content
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except IOError as e:
            self.logger.error(f"Failed to read code from {file_path}: {e}")
            return ""

    def __init__(self, config: Config):
        """
        Initialize the Engine with the given configuration.

        Args:
            config (Config): Configuration parameters.
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config
        self.planning_method = config.engine_planner.get("planning_method", "naive")
        # Initialize Environment
        self.environment = self._initialize_environment(config.environment)
        # Initialize Agents
        self.agents = self._initialize_agents(config.agents)
        # Initialize AgentGraph
        self.graph = AgentGraph(self.agents, config)
        for agent in self.agents:
            agent.set_agent_graph(self.graph)
        # Initialize Memory
        self.memory = self._initialize_memory(config.memory)
        # Initialize Evaluator
        self.evaluator = Evaluator(metrics_config=config.metrics)
        self.task = config.task.get("content", "")
        if isinstance(self.environment, CodingEnvironment):
            self.environment.task_description = self.task
            self.environment.state["task_description"] = self.task
        self.output_format = config.task.get(
            "output_format",
            "You are free to define your own output format to answer the task properly.",
        )
        self.coordinate_mode = config.coordination_mode
        # Initialize EnginePlanner
        self.planner = EnginePlanner(
            agent_graph=self.graph,
            memory=self.memory,
            config=config.engine_planner,
            task=self.task,
            model=config.llm,
        )
        self.max_iterations = config.environment.get("max_iterations", 10)
        self.current_iteration = 0

        self.logger.info("Engine initialized.")

    def _initialize_environment(self, env_config: Dict[str, Any]) -> BaseEnvironment:
        """
        Initialize the environment based on configuration.

        Args:
            env_config (dict): Environment configuration.

        Returns:
            BaseEnvironment: An instance of the environment.

        Raises:
            ValueError: If the environment type is not supported.
        """
        env_type = env_config.get("type")

        if env_type == "Web":
            env1 = WebEnvironment(name="Web Environment", config=env_config)
            return env1
        elif env_type == "Base":
            env2 = BaseEnvironment(name="Base Environment", config=env_config)
            return env2
        elif env_type == "Research":
            env3 = ResearchEnvironment(name="Research Environment", config=env_config)
            return env3
        elif env_type == "Coding":
            env4 = CodingEnvironment(name="Coding Environment", config=env_config)
            return env4
        elif env_type == "WorldSimulation":
            env4 = WorldSimulationEnvironment(
                name="World Simulation Environment", config=env_config
            )
            return env4
        elif env_type == "Minecraft":
            env5 = MinecraftEnvironment(name="Minecraft Environment", config=env_config)
            return env5
        elif env_type == "DB":
            env6 = DBEnvironment(name="DB Environment", config=env_config)
            return env6
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")

    def _initialize_agents(
        self, agent_configs: List[Dict[str, Any]]
    ) -> List[BaseAgent]:
        """
        Initialize agents based on configurations.

        Args:
            agent_configs (List[dict]): List of agent configurations.

        Returns:
            List[BaseAgent]: List of agent instances.
        """
        agents = []
        llm = self.config.llm
        for agent_config in agent_configs:
            agent_llm = agent_config.get(
                "llm", llm
            )  # use agent-specific LLM if provided
            agent_type = agent_config.get("type")
            agent = BaseAgent(
                config=agent_config, env=self.environment, model=agent_llm
            )
            agents.append(agent)
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' using LLM '{agent_llm}' initialized."
            )
            if isinstance(self.environment, MinecraftEnvironment):
                assert "agent_id" in agent_config and "agent_port" in agent_config
                self.environment.register_agent(
                    agent_config.get("agent_id"), agent_config.get("agent_port")
                )
            self.logger.debug(
                f"Agent '{agent.agent_id}' of type '{agent_type}' initialized."
            )
        return agents

    def _initialize_memory(
        self, memory_config: Dict[str, Any]
    ) -> Union[SharedMemory, BaseMemory]:
        """
        Initialize the shared memory mechanism.

        Args:
            memory_config (dict): Memory configuration.

        Returns:
            BaseMemory: An instance of the memory module.
        """
        memory_type = memory_config.get("type", "SharedMemory")
        memory: Union[BaseMemory, SharedMemory, None] = None
        if memory_type == "SharedMemory":
            memory = SharedMemory()
        else:
            memory = BaseMemory()
        self.logger.debug(f"Memory of type '{memory_type}' initialized.")
        return memory

    def graph_coordinate(self) -> None:
        """
        Graph-based coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Initial assignment: Distribute the overall task to each agent
            self.logger.info("Initial task distribution to all agents.")
            initial_tasks = {
                agent.agent_id: self.task for agent in self.graph.get_all_agents()
            }
            agents_results = []

            # Initialize iteration_data for the initial assignment to match iterative structure
            iteration_data = {
                "iteration": self.current_iteration + 1,
                "task_assignments": {},
                "task_results": [],
                "summary": "",
                "continue_simulation": True,
                "communications": [],
            }
            communications = []
            for agent_id, task in initial_tasks.items():
                try:
                    agent = self.graph.get_agent(agent_id)
                    self.logger.info(f"Assigning initial task to {agent_id}: {task}")
                    # Assign the task to the agent
                    iteration_data_task_assignments = iteration_data.get(
                        "task_assignments"
                    )
                    assert isinstance(iteration_data_task_assignments, dict)
                    iteration_data_task_assignments[agent_id] = task
                    result, communication = agent.act(task)
                    self.logger.info(f"Processing result for agent '{agent.agent_id}'")
                    self.logger.info(f"Communication received: {communication}")
                    if communication:
                        self.logger.info(
                            f"Adding communication to list: {communication}"
                        )
                        communications.append(communication)
                    agents_results.append({agent_id: result})
                    # Record the result
                    task_result = {"agent_id": agent_id, "result": result}
                    iteration_data_task_results = iteration_data.get("task_results")
                    assert isinstance(iteration_data_task_results, list)
                    iteration_data_task_results.append(task_result)
                    self.logger.debug(
                        f"Agent '{agent_id}' completed initial task with result: {result}"
                    )
                except KeyError:
                    self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                except Exception as e:
                    self.logger.error(
                        f"Error while executing initial task for agent '{agent_id}': {e}"
                    )
            iteration_data["communications"] = communications
            # Summarize outputs and update planner for the initial assignment
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Initial Summary:\n{summary}")
            summary = self.planner.summarize_output(
                summary, self.task, self.output_format
            )
            iteration_data["summary"] = summary.content

            # Decide whether to continue or terminate after initial assignment
            if isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                self.logger.info(
                    f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                )
                continue_simulation = int(block_hit_rate) != 1
            else:
                continue_simulation = self.planner.decide_next_step(agents_results)
            iteration_data["continue_simulation"] = continue_simulation
            if not continue_simulation:
                self.logger.info(
                    "EnginePlanner decided to terminate the simulation after initial assignment."
                )
            else:
                self.planner.update_progress(summary)
                self.current_iteration += 1

            summary_data["iterations"].append(iteration_data)

            # Evaluate communication (enabled)
            if iteration_data["communications"]:
                iteration_data_communications = iteration_data.get("communications")
                assert isinstance(iteration_data_communications, list)
                communications_str = self._format_communications(
                    iteration_data_communications
                )
                self.evaluator.evaluate_communication(self.task, communications_str)
            else:
                self.logger.info("No communications to evaluate")
                # Store -1 if communications are empty
                self.evaluator.metrics["communication_score"].append(-1)

            # Evaluate planning + KPI (enabled)
            agent_profiles = self._get_agent_profiles()
            iteration_data_task_assignments = iteration_data.get("task_assignments")
            assert isinstance(iteration_data_task_assignments, dict)
            agent_tasks_str = self._format_agent_tasks(iteration_data_task_assignments)
            iteration_data_task_results = iteration_data.get("task_results")
            assert isinstance(iteration_data_task_results, list)
            results_str = self._format_results(iteration_data_task_results)
            iteration_data_summary = iteration_data.get("summary")
            assert isinstance(iteration_data_summary, str)
            self.evaluator.evaluate_planning(
                iteration_data_summary, agent_profiles, agent_tasks_str, results_str
            )
            self.evaluator.evaluate_kpi(self.task, results_str)

            end_on_iter_0 = False
            if not continue_simulation:
                end_on_iter_0 = True

            while self.current_iteration < self.max_iterations and not end_on_iter_0:
                iteration_data = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "communications": [],
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                current_agents = self.graph.get_all_agents()
                current_tasks = {}
                agents_results = []
                communications = []

                for agent in current_agents:
                    try:
                        # Each agent plans its own task
                        task = agent.plan_task()
                        current_tasks[agent.agent_id] = task
                        iteration_data_task_assignments = iteration_data.get(
                            "task_assignments"
                        )
                        assert isinstance(iteration_data_task_assignments, dict)
                        iteration_data_task_assignments[agent.agent_id] = task
                        self.logger.info(
                            f"Agent '{agent.agent_id}' planned task: {task}"
                        )

                        # Agent acts on the planned task
                        result, communication = agent.act(task)
                        self.logger.info(
                            f"Processing result for agent '{agent.agent_id}'"
                        )
                        self.logger.info(f"Communication received: {communication}")
                        if communication:
                            self.logger.info(
                                f"Adding communication to list: {communication}"
                            )
                            communications.append(communication)
                        agents_results.append({agent.agent_id: result})
                        iteration_data_task_results = iteration_data.get("task_results")
                        assert isinstance(iteration_data_task_results, list)
                        iteration_data_task_results.append({agent.agent_id: result})
                        self.logger.debug(
                            f"Agent '{agent.agent_id}' executed task with result: {result}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error in agent '{agent.agent_id}' during planning or action: {e}"
                        )
                iteration_data["communications"] = communications
                # Summarize outputs and update planner
                summary = self._summarize_results(agents_results)
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.current_iteration += 1
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate communication (enabled)
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    communications_str = self._format_communications(
                        iteration_data_communications
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    self.logger.info("No communications to evaluate")
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning + KPI (enabled)
                agent_profiles = self._get_agent_profiles()
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data_task_assignments
                )
                iteration_data_task_results = iteration_data.get("task_results")
                assert isinstance(iteration_data_task_results, list)
                results_str = self._format_results(iteration_data_task_results)
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_planning(
                    iteration_data_summary, agent_profiles, agent_tasks_str, results_str
                )
                self.evaluator.evaluate_kpi(self.task, results_str)
                # Decide whether to continue or terminate
                if isinstance(self.environment, MinecraftEnvironment):
                    try:
                        with open("../data/score.json", "r") as f:
                            block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                    except:
                        block_hit_rate = 0.0
                    self.logger.info(
                        f"Using a rule-based EnginePlanner. block_hit_rate is {block_hit_rate}"
                    )
                    continue_simulation = int(block_hit_rate) != 1
                else:
                    continue_simulation = self.planner.decide_next_step(agents_results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                # # Check if task is completed within the environment
                # if self.environment.is_task_completed():
                #     self.logger.info("Task has been completed successfully.")
                #     break
            # At the end, add the scores to summary_data

            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            # if self.environment.name == 'Research Environment':
            if isinstance(self.environment, ResearchEnvironment):
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_task_research(self.task, iteration_data_summary)
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            elif isinstance(self.environment, MinecraftEnvironment):
                try:
                    with open("../data/score.json", "r") as f:
                        block_hit_rate = json.load(f)[-1]["block_hit_rate"]
                except:
                    block_hit_rate = 0.0
                summary_data["task_evaluation"] = block_hit_rate * 5
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            self.logger.info("Engine graph-based coordination loop completed.")

        except Exception:
            self.logger.exception("An error occurred during graph-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Graph-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def star_coordinate(self) -> None:
        """
        Centralized coordination mode.
        """
        try:
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
                "final_output": "",
            }
            agents_results: List[Dict[str, Any]] = []
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "task_assignments": {},
                    "task_results": [],
                    "summary": "",
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Starting iteration {self.current_iteration}")

                # Assign tasks to agents
                assignment = self.planner.assign_tasks(
                    planning_method=self.planning_method
                )
                tasks = assignment.get("tasks", {})
                iteration_data["task_assignments"] = tasks
                self.logger.info(f"Assigned tasks: {tasks}")

                # Assign tasks to agents
                agents_results = []
                communications = []
                for agent_id, task in tasks.items():
                    try:
                        agent = self.graph.get_agent(agent_id)
                        self.logger.info(f"Assigning task to {agent_id}: {task}")
                        result, communication = agent.act(task)
                        agents_results.append({agent_id: result})
                        if communication:
                            communications.append(communication)

                        self.logger.debug(
                            f"Agent '{agent_id}' completed task with result: {result}"
                        )
                    except KeyError:
                        self.logger.error(f"Agent '{agent_id}' not found in the graph.")
                    except Exception as e:
                        self.logger.error(
                            f"Error while executing task for agent '{agent_id}': {e}"
                        )
                iteration_data["task_results"] = agents_results
                iteration_data["communications"] = communications
                # Update progress based on agents' results
                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content
                self.logger.info(summary)
                self.planner.update_progress(summary)
                self.current_iteration += 1

                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(agents_results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break

                if self.current_iteration >= self.max_iterations:
                    self.logger.info("Maximum iterations reached.")
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if self.environment.name == "Coding Environment":
                code = self._read_code_from_file("MARBLE/marble/workspace/solution.py")
                if code:
                    self.evaluator.evaluate_code_quality(
                        task=self.task, code_result=code
                    )
                    summary_data["code_quality"] = self.evaluator.metrics[
                        "code_quality"
                    ]
                    self.logger.info(
                        f"Code quality evaluation results: {self.evaluator.metrics['code_quality']}"
                    )
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine star-based coordination loop completed.")
            self.logger.info("Engine simulation loop completed.")

        except Exception:
            self.logger.exception("An error occurred during simulation.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Simulation completed.")
            self._write_to_jsonl(summary_data)

    def chain_coordinate(self) -> None:
        """
        Chain-based coordination mode.
        """
        try:
            self.logger.info("Starting chain-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }
            # Start with the initial agent
            current_agent = self._select_initial_agent()
            if not current_agent:
                self.logger.error("No initial agent found for chain.")
                return

            max_chain_length = self.max_iterations * len(
                self.agents
            )  # Or define a separate chain length limit
            chain_length = 0

            task = self.task
            agents_results = []

            while current_agent and chain_length < max_chain_length:
                iteration_data = {
                    "chain_length": chain_length + 1,
                    "current_agent": current_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "task_assignments": {},
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.logger.info(f"Agent '{current_agent.agent_id}' is executing task.")
                result, communication = current_agent.act(task)
                result_str = f"AgentID: '{current_agent.agent_id}' completed task with result: {result}"
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                iteration_data_task_assignments[current_agent.agent_id] = task
                agents_results.append({current_agent.agent_id: result})
                iteration_data["result"] = result
                self.logger.info(
                    f"Agent '{current_agent.agent_id}' completed task with result: {result}"
                )
                # Get profiles of other agents
                agent_profiles = self.graph.get_agent_profiles_linked(
                    current_agent.agent_id
                )
                # Current agent chooses the next agent
                next_agent_id, plan = current_agent.plan_next_agent(
                    result, agent_profiles
                )
                current_agent_ = current_agent
                try:
                    current_agent = self.graph.get_agent(next_agent_id)
                except Exception:
                    self.logger.error(
                        f"Agent '{next_agent_id}' not found in the graph. keep the same agent."
                    )
                    current_agent = current_agent_
                task = plan
                chain_length += 1
                self.planner.update_progress(result)
                iteration_data["communications"] = communication

                # Evaluate communication
                if iteration_data["communications"]:
                    iteration_data_communications = iteration_data.get("communications")
                    assert isinstance(iteration_data_communications, list)
                    communications_str = self._format_communications(
                        iteration_data_communications
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                summary = self._summarize_results(agents_results)
                summary_from_planner = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary_from_planner.content

                # Evaluate planning
                agent_profiles_self = self._get_agent_profiles()
                iteration_data_task_assignments = iteration_data.get("task_assignments")
                assert isinstance(iteration_data_task_assignments, dict)
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data_task_assignments
                )
                iteration_data_summary = iteration_data.get("summary")
                assert isinstance(iteration_data_summary, str)
                self.evaluator.evaluate_planning(
                    iteration_data_summary, agent_profiles_self, agent_tasks_str, result
                )
                self.evaluator.evaluate_kpi(self.task, result_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(
                    [{"root_agent": result}]
                )
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # Update progress
            summary = self._summarize_results(agents_results)
            self.logger.info(f"Chain execution Summary:\n{summary}")
            self.planner.update_progress(summary)

            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                # summary_data['task_evaluation'] = self.evaluator.metrics["task_evaluation"]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine chain-based coordination loop completed.")
            self.logger.info("Chain-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during chain-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Chain-based coordination simulation completed.")
            summary_data["token_usage"] = self._get_totoal_token_usage()
            self._write_to_jsonl(summary_data)

    def tree_coordinate(self) -> None:
        """
        Tree-based coordination mode.
        """
        try:
            self.logger.info("Starting tree-based coordination.")
            summary_data = {
                "task": self.task,
                "coordination_mode": self.coordinate_mode,
                "iterations": [],
            }

            root_agent = self.graph.get_root_agent()
            if not root_agent:
                self.logger.error("No root agent found in the tree.")
                return
            # Start the coordination from the root agent
            while self.current_iteration < self.max_iterations:
                iteration_data: Dict[str, Any] = {
                    "iteration": self.current_iteration + 1,
                    "root_agent": root_agent.agent_id,
                    "result": None,
                    "continue_simulation": True,
                    "total_milestones": 0,
                    "agent_kpis": {},
                }
                self.current_iteration += 1
                self.logger.info(f"Starting iteration {self.current_iteration}")
                results, communication, tasks = self._execute_agent_task_recursive(
                    root_agent, self.task
                )
                # Update progress
                summary = self._summarize_results(results)
                summary = self.planner.summarize_output(
                    summary, self.task, self.output_format
                )
                iteration_data["summary"] = summary.content
                self.logger.info(
                    f"Iteration {self.current_iteration} Summary:\n{summary}"
                )
                self.planner.update_progress(summary)
                iteration_data["communications"] = communication
                iteration_data["task_assignments"] = tasks
                iteration_data["task_results"] = results
                # Evaluate communication
                if iteration_data["communications"]:
                    communications_str = self._format_communications(
                        iteration_data["communications"]
                    )
                    self.evaluator.evaluate_communication(self.task, communications_str)
                else:
                    # Store -1 if communications are empty
                    self.evaluator.metrics["communication_score"].append(-1)

                # Evaluate planning
                agent_profiles = self._get_agent_profiles()
                agent_tasks_str = self._format_agent_tasks(
                    iteration_data["task_assignments"]
                )
                results_str = self._format_results(iteration_data["task_results"])
                self.evaluator.evaluate_planning(
                    iteration_data["summary"],
                    agent_profiles,
                    agent_tasks_str,
                    results_str,
                )
                self.evaluator.evaluate_kpi(self.task, results_str)

                # Decide whether to continue or terminate
                continue_simulation = self.planner.decide_next_step(results)
                iteration_data["continue_simulation"] = continue_simulation
                summary_data["iterations"].append(iteration_data)
                if not continue_simulation:
                    self.logger.info(
                        "EnginePlanner decided to terminate the simulation."
                    )
                    break
            # At the end, add the scores to summary_data
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics[
                "total_milestones"
            ]
            if self.environment.name == "Research Environment":
                self.evaluator.evaluate_task_research(
                    self.task, iteration_data["summary"]
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine graph-based coordination loop completed.")
            if self.environment.name == "Coding Environment":
                code = self._read_code_from_file("MARBLE/marble/workspace/solution.py")
                if code:
                    self.evaluator.evaluate_code_quality(
                        task=self.task, code_result=code
                    )
                    summary_data["code_quality"] = self.evaluator.metrics[
                        "code_quality"
                    ]
                    self.logger.info(
                        f"Code quality evaluation results: {self.evaluator.metrics['code_quality']}"
                    )
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, iteration_data["summary"])
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            elif self.environment.name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    iteration_data["summary"],
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
                summary_data["task_evaluation"] = self.evaluator.metrics[
                    "task_evaluation"
                ]
                self.logger.info("Engine tree-based coordination loop completed.")
            self.logger.info("Tree-based coordination simulation completed.")

        except Exception:
            self.logger.exception("An error occurred during tree-based coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self.logger.info("Tree-based coordination simulation completed.")
            self._write_to_jsonl(summary_data)

    def _execute_agent_task_recursive(self, agent: BaseAgent, task: str) -> Any:
        """
        Recursively execute tasks starting from the given agent.

        Args:
            agent (BaseAgent): The agent to execute task.
            task (str): The task to execute.

        Returns:
            Any: The result of the agent's execution.
        """
        self.logger.info(f"Agent '{agent.agent_id}' is executing task.")
        tasks = []
        print(agent.children)
        if len(agent.children) > 0:
            print("******************start recursive******************")
            # Agent assigns tasks to children
            tasks_for_children = agent.plan_tasks_for_children(task)
            tasks.append(tasks_for_children)
            children_results = []
            communications = []
            for child in agent.children:
                child_task = tasks_for_children.get(child.agent_id, "")
                if child_task:
                    (
                        child_result,
                        communication,
                        tasks_,
                    ) = self._execute_agent_task_recursive(child, child_task)
                    tasks += tasks_
                    if communication:
                        communications.append(communication)
                    children_results += child_result
            # Agent may also act itself
            results_str = "\n".join(
                json.dumps(result)[:500] for result in children_results
            )

            task_for_father = (
                task
                + "\nHere are the results of the children: "
                + results_str
                + "\nPlease don't repeat the same task and continue to work on the original task. You may also need to communicate with other agents or summarize the results or just continue to work on the original task."
            )
            own_result, communication = agent.act(task_for_father)

            if communication:
                communications.append(communication)
            communications_str = "\n".join(communications) if communications else None
            # # Combine results
            # combined_result = agent.summarize_results(children_results, own_result)
            results = [
                {"agent_id": agent.agent_id, "result": own_result}
            ] + children_results
            return results, communications_str, tasks
        else:
            # Agent directly acts on the task
            result, communication = agent.act(task)
            return (
                [{"agent_id": agent.agent_id, "result": result}],
                communication,
                tasks,
            )

    def _select_initial_agent(self) -> Optional[BaseAgent]:
        """
        Select the initial agent to start the chain.

        Returns:
            Optional[BaseAgent]: The initial agent, or None if not found.
        """
        # For simplicity, select an agent based on some criteria.
        # Here, we'll select the agent with the highest priority or a predefined agent.
        # Alternatively, we could prompt the LLM to select the starting agent.

        # Example: Select agent1 as the starting agent
        starting_agent_id = "agent1"
        if starting_agent_id in [agent.agent_id for agent in self.agents]:
            return self.graph.get_agent(starting_agent_id)
        else:
            self.logger.error(f"Starting agent '{starting_agent_id}' not found.")
            return None

    def latent_coordinate(self) -> None:
        """
        LatentMAS-style coordination: agents communicate in latent space.

        Reuses the vendored LatentMAS ``ModelWrapper`` (see
        ``marble.llms.latent_mas_model``). A single KV cache (the shared latent
        working memory) is threaded sequentially through the agents in config
        order: each non-final agent prefills its profile/task prompt on top of
        the accumulated cache and appends ``latent_steps`` continuous thoughts
        via :meth:`ModelWrapper.generate_latent_batch`; the final agent acts as
        the judger and decodes the answer from that cache with
        :meth:`ModelWrapper.generate_text_batch`. No text is exchanged between
        agents, and every call runs on the local white-box model.
        """
        import torch

        from marble.llms.latent_mas_model import (
            ModelWrapper,
            _past_length,
            truncate_past,
        )

        cfg = self.config.latent or {}
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        latent_steps = int(cfg.get("latent_steps", 10))
        max_new_tokens = int(cfg.get("max_new_tokens", 512))
        temperature = float(cfg.get("temperature", 0.7))
        top_p = float(cfg.get("top_p", 0.95))
        latent_space_realign = bool(cfg.get("latent_space_realign", False))
        # Optional context bounding, mirroring upstream LatentMAS flags.
        latent_only = bool(cfg.get("latent_only", False))
        sequential_info_only = bool(cfg.get("sequential_info_only", False)) or latent_only
        device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        summary_data: Dict[str, Any] = {
            "task": self.task,
            "coordination_mode": self.coordinate_mode,
            "communication_mode": "latent_mas",
            "model_name": model_name,
            "latent_steps": latent_steps,
            "agents": [],
            "final_output": "",
        }
        try:
            self.logger.info(
                f"Loading latent model '{model_name}' on {device} "
                f"(latent_steps={latent_steps}, realign={latent_space_realign})."
            )
            model = ModelWrapper(
                model_name, device, latent_space_realign=latent_space_realign
            )

            agents = self.graph.get_all_agents()
            if not agents:
                self.logger.error("No agents found for latent coordination.")
                return

            past_kv: Optional[Any] = None
            total_latent_steps = 0
            final_output = ""

            for index, agent in enumerate(agents):
                is_last = index == len(agents) - 1
                messages = self._build_latent_messages(
                    agent, has_context=past_kv is not None, is_judger=is_last
                )
                _, input_ids, attention_mask, _ = model.prepare_chat_batch([messages])

                if not is_last:
                    prev_len = _past_length(past_kv)
                    past_kv = model.generate_latent_batch(
                        input_ids,
                        attention_mask=attention_mask,
                        latent_steps=latent_steps,
                        past_key_values=past_kv,
                    )
                    if sequential_info_only:
                        added = _past_length(past_kv) - prev_len
                        keep = latent_steps if latent_only else added
                        past_kv = truncate_past(past_kv, keep)
                    total_latent_steps += latent_steps
                    summary_data["agents"].append(
                        {"agent_id": agent.agent_id, "role": "latent", "output": ""}
                    )
                    self.logger.info(
                        f"Agent '{agent.agent_id}' produced {latent_steps} latent thoughts."
                    )
                else:
                    generations, _ = model.generate_text_batch(
                        input_ids,
                        attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        past_key_values=past_kv if latent_steps > 0 else None,
                    )
                    final_output = generations[0].strip()
                    summary_data["agents"].append(
                        {
                            "agent_id": agent.agent_id,
                            "role": "judger",
                            "output": final_output,
                        }
                    )
                    self.logger.info(
                        f"Judger '{agent.agent_id}' decoded final answer."
                    )

            summary_data["final_output"] = final_output
            summary_data["total_latent_steps"] = total_latent_steps
            summary_data["decoded_tokens"] = int(
                model.tokenize_text(final_output).shape[-1]
            ) if final_output else 0
            self.logger.info(f"Latent coordination final output:\n{final_output}")

            # Score planning/KPI/task with the original evaluator (see
            # _evaluate_latent); communication stays N/A for latent.
            self._evaluate_latent(summary_data, final_output, agents)

        except Exception:
            self.logger.exception("An error occurred during latent coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self._write_to_jsonl(summary_data)
            self.logger.info("Latent coordination simulation completed.")

    def _evaluate_latent(
        self, summary_data: Dict[str, Any], final_output: str, agents: List[BaseAgent]
    ) -> None:
        """
        Score a latent run, reusing the original evaluator like chain/tree do.

        - Planning + KPI: scored with the original LLM-judge methods, so latent
          is comparable to the text baseline on these metrics.
        - Communication: not applicable -- the MultiAgentBench communication
          score (arXiv:2503.01935 §3.3) rates inter-agent *text* messages, which
          latent (KV) communication does not produce. Left unscored.
        - Coordination = planning only for latent (communication is N/A).
        - All judge calls are isolated so a failure never drops the final output.
        """
        # Planning via the original LLM judge (holistic rating from the final
        # output + profiles). KPI is intentionally NOT scored for latent: the
        # milestone-based KPI needs each agent's text contribution to attribute
        # milestones, but latent agents (except the final one) emit only KV, so
        # the judge would have to fabricate per-agent attribution.
        try:
            agent_profiles = self._get_agent_profiles()
            agent_tasks = self._format_agent_tasks(
                {agent.agent_id: self.task for agent in agents}
            )
            self.evaluator.evaluate_planning(
                final_output, agent_profiles, agent_tasks, final_output
            )
        except Exception:
            self.logger.exception("Latent planning evaluation failed.")
            self.evaluator.metrics["planning_score"].append(-1)

        planning_scores = self.evaluator.metrics["planning_score"]
        valid_planning = [s for s in planning_scores if s is not None and s >= 0]

        summary_data["planning_scores"] = planning_scores
        summary_data["communication_scores"] = []
        summary_data["communication_note"] = (
            "not scored: the communication score rates inter-agent text "
            "messages, which latent (KV) communication does not produce."
        )
        summary_data["coordination_score"] = (
            sum(valid_planning) / len(valid_planning) if valid_planning else None
        )
        summary_data["token_usage"] = self._get_totoal_token_usage()

        # Per-environment task score reuses the original evaluator (same dispatch
        # as chain/tree); isolated so a failure can't drop the final output.
        try:
            env_name = self.environment.name
            if env_name == "Research Environment":
                self.evaluator.evaluate_task_research(self.task, final_output)
            elif env_name == "World Simulation Environment":
                self.evaluator.evaluate_task_world(self.task, final_output)
            elif env_name == "DB Environment":
                self.evaluator.evaluate_task_db(
                    self.task,
                    final_output,
                    self.config.task["labels"],
                    self.config.task["number_of_labels_pred"],
                    self.config.task["root_causes"],
                )
        except Exception:
            self.logger.exception("Latent task evaluation failed; leaving it empty.")

        # KPI/milestone attribution is not measurable for latent (see above).
        summary_data["agent_kpis"] = {}
        summary_data["total_milestones"] = None
        summary_data["kpi_note"] = (
            "not scored: milestone KPI / per-agent attribution needs each agent's "
            "text output; latent agents (except the final one) produce only KV."
        )
        summary_data["task_evaluation"] = self.evaluator.metrics["task_evaluation"]

    def _build_latent_messages(
        self, agent: BaseAgent, has_context: bool, is_judger: bool
    ) -> List[Dict[str, str]]:
        """Build the chat messages for one agent in the latent chain."""
        profile = agent.get_profile()
        if is_judger:
            context_note = (
                "You are given the previous agents' latent working memory for "
                "reference; it may contain irrelevant content, so use it only if "
                "helpful.\n"
                if has_context
                else ""
            )
            user = (
                f"You are {agent.agent_id}: {profile}\n"
                f"Task: {self.task}\n"
                f"{context_note}"
                "Now solve the task and produce the final answer.\n"
                f"Output format: {self.output_format}"
            )
        else:
            context_note = (
                "The previous agents' latent working memory is already attended "
                "to in your context; build on it.\n"
                if has_context
                else ""
            )
            user = (
                f"You are {agent.agent_id}: {profile}\n"
                f"Task: {self.task}\n"
                f"{context_note}"
                "Contribute your reasoning toward solving the task. "
                "Do not produce the final answer."
            )
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ]

    def fixed_chain_coordinate(self) -> None:
        """
        Fixed-pipeline TEXT baseline, structurally mirroring latent_coordinate.

        Every agent runs once in config order (no dynamic routing or early stop,
        unlike chain_coordinate); each agent's TEXT output is passed to the next
        as context -- this is the text communication channel. The last agent
        produces the final answer. This is the apples-to-apples text counterpart
        of the latent baseline: same agents, same order, only the communication
        channel differs (plaintext vs latent KV).
        """
        from litellm.utils import token_counter

        from marble.llms.model_prompting import model_prompting

        summary_data: Dict[str, Any] = {
            "task": self.task,
            "coordination_mode": self.coordinate_mode,
            "communication_mode": "text",
            "agents": [],
            "final_output": "",
        }
        try:
            agents = self.graph.get_all_agents()
            if not agents:
                self.logger.error("No agents found for fixed-chain coordination.")
                return

            context = ""
            handoffs: List[str] = []
            final_output = ""
            for index, agent in enumerate(agents):
                is_last = index == len(agents) - 1
                messages = self._build_chain_text_messages(agent, context, is_last)
                result = model_prompting(
                    llm_model=agent.llm,
                    messages=messages,
                    return_num=1,
                    max_token_num=512,
                    temperature=0.0,
                    top_p=None,
                    stream=None,
                )[0]
                text = result.content or ""
                agent.token_usage += token_counter(
                    model=agent.llm,
                    messages=messages + [{"role": "assistant", "content": text}],
                )
                summary_data["agents"].append(
                    {
                        "agent_id": agent.agent_id,
                        "role": "judger" if is_last else "agent",
                        "output": text,
                    }
                )
                if is_last:
                    final_output = text
                else:
                    handoff = f"From {agent.agent_id}: {text}"
                    context += handoff + "\n"
                    handoffs.append(handoff)
                self.logger.info(
                    f"Agent '{agent.agent_id}' produced output (is_last={is_last})."
                )
            summary_data["final_output"] = final_output

            # Scoring with the original evaluator: communication on the text
            # handoffs (this is what latent has N/A), plus planning/KPI/task.
            communications_str = "\n".join(handoffs)
            if communications_str.strip():
                self.evaluator.evaluate_communication(self.task, communications_str)
            else:
                self.evaluator.metrics["communication_score"].append(-1)
            try:
                agent_profiles = self._get_agent_profiles()
                agent_tasks = self._format_agent_tasks(
                    {agent.agent_id: self.task for agent in agents}
                )
                # Feed the judge each agent's labeled output so KPI can attribute
                # milestones to the right agent IDs (not just the final answer).
                agent_results_str = "\n".join(
                    f"From {a['agent_id']}: {a['output']}"
                    for a in summary_data["agents"]
                )
                self.evaluator.evaluate_planning(
                    final_output, agent_profiles, agent_tasks, agent_results_str
                )
                self.evaluator.evaluate_kpi(self.task, agent_results_str)
            except Exception:
                self.logger.exception("Fixed-chain planning/KPI evaluation failed.")
                self.evaluator.metrics["planning_score"].append(-1)
            try:
                env_name = self.environment.name
                if env_name == "Research Environment":
                    self.evaluator.evaluate_task_research(self.task, final_output)
                elif env_name == "World Simulation Environment":
                    self.evaluator.evaluate_task_world(self.task, final_output)
                elif env_name == "DB Environment":
                    self.evaluator.evaluate_task_db(
                        self.task,
                        final_output,
                        self.config.task["labels"],
                        self.config.task["number_of_labels_pred"],
                        self.config.task["root_causes"],
                    )
            except Exception:
                self.logger.exception(
                    "Fixed-chain task evaluation failed; leaving it empty."
                )

            planning = self.evaluator.metrics["planning_score"]
            comm = self.evaluator.metrics["communication_score"]
            valid_p = [s for s in planning if s is not None and s >= 0]
            valid_c = [s for s in comm if s is not None and s >= 0]
            summary_data["planning_scores"] = planning
            summary_data["communication_scores"] = comm
            # Coordination = (planning + communication)/2 (paper formula; the
            # text baseline has both sub-scores, unlike latent which is planning-only).
            p_avg = sum(valid_p) / len(valid_p) if valid_p else None
            c_avg = sum(valid_c) / len(valid_c) if valid_c else None
            if p_avg is not None and c_avg is not None:
                summary_data["coordination_score"] = (p_avg + c_avg) / 2
            else:
                summary_data["coordination_score"] = p_avg if c_avg is None else c_avg
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics["total_milestones"]
            summary_data["task_evaluation"] = self.evaluator.metrics["task_evaluation"]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            self.logger.info(f"Fixed-chain final output:\n{final_output}")
        except Exception:
            self.logger.exception("An error occurred during fixed-chain coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self._write_to_jsonl(summary_data)
            self.logger.info("Fixed-chain coordination simulation completed.")

    def _build_chain_text_messages(
        self, agent: BaseAgent, context: str, is_judger: bool
    ) -> List[Dict[str, str]]:
        """Messages for one agent in the fixed TEXT chain (text communication)."""
        profile = agent.get_profile()
        ctx = f"Previous agents' outputs:\n{context}\n" if context.strip() else ""
        if is_judger:
            user = (
                f"You are {agent.agent_id}: {profile}\n"
                f"Task: {self.task}\n"
                f"{ctx}"
                "Now solve the task and produce the final answer.\n"
                f"Output format: {self.output_format}"
            )
        else:
            user = (
                f"You are {agent.agent_id}: {profile}\n"
                f"Task: {self.task}\n"
                f"{ctx}"
                "Contribute your reasoning toward solving the task. "
                "Do not produce the final answer."
            )
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ]

    def graph_latent_coordinate(self) -> None:
        """
        Graph-style latent coordination (B1, no env tools).

        Reuses LatentMAS sequential KV threading (NOT a merge): per round, one KV
        cache is threaded through every worker agent (latent reasoning), then the
        judger agent decodes the round's answer from the accumulated KV -- this
        replaces MARBLE's text summarize_output. A decoded true/false
        (ModelWrapper.decode_bool) replaces decide_next_step as the stop signal.
        No agent<->tool channel: the judger decodes the answer directly, so this
        fits tool-free or code-generation tasks (the judger writes the code);
        external-info tools (db/research/minecraft) need the future B2 path.
        """
        import torch

        from marble.llms.latent_mas_model import (
            ModelWrapper,
            _past_length,
            truncate_past,
        )

        cfg = self.config.latent or {}
        model_name = cfg.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        latent_steps = int(cfg.get("latent_steps", 10))
        max_new_tokens = int(cfg.get("max_new_tokens", 512))
        temperature = float(cfg.get("temperature", 0.7))
        top_p = float(cfg.get("top_p", 0.95))
        latent_space_realign = bool(cfg.get("latent_space_realign", False))
        latent_only = bool(cfg.get("latent_only", False))
        sequential_info_only = (
            bool(cfg.get("sequential_info_only", False)) or latent_only
        )
        device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        # Same top-level schema as graph_coordinate (iterations + the standard
        # metric fields), plus latent-only extras appended at the end.
        summary_data: Dict[str, Any] = {
            "task": self.task,
            "coordination_mode": self.coordinate_mode,
            "communication_mode": "latent_mas",
            "model_name": model_name,
            "latent_steps": latent_steps,
            "iterations": [],
        }
        try:
            model = ModelWrapper(
                model_name, device, latent_space_realign=latent_space_realign
            )
            agents = self.graph.get_all_agents()
            if not agents:
                self.logger.error("No agents found for graph-latent coordination.")
                return
            # Last agent is the judger/summarizer; the rest are workers.
            workers = agents[:-1] if len(agents) > 1 else agents
            judger = agents[-1]
            agent_profiles = self._get_agent_profiles()
            agent_tasks_str = self._format_agent_tasks(
                {agent.agent_id: self.task for agent in agents}
            )

            past_kv: Optional[Any] = None
            total_latent_steps = 0
            final_output = ""
            for rnd in range(max(1, self.max_iterations)):
                # Workers: thread ONE KV through them (latent reasoning).
                for agent in workers:
                    messages = self._build_latent_messages(
                        agent, has_context=past_kv is not None, is_judger=False
                    )
                    _, ids, mask, _ = model.prepare_chat_batch([messages])
                    prev_len = _past_length(past_kv)
                    past_kv = model.generate_latent_batch(
                        ids,
                        attention_mask=mask,
                        latent_steps=latent_steps,
                        past_key_values=past_kv,
                    )
                    if sequential_info_only:
                        added = _past_length(past_kv) - prev_len
                        keep = latent_steps if latent_only else added
                        past_kv = truncate_past(past_kv, keep)
                    total_latent_steps += latent_steps
                # Judger: decode this round's answer from the accumulated KV.
                jmsg = self._build_latent_messages(
                    judger, has_context=past_kv is not None, is_judger=True
                )
                _, jids, jmask, _ = model.prepare_chat_batch([jmsg])
                gens, _ = model.generate_text_batch(
                    jids,
                    jmask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    past_key_values=past_kv if latent_steps > 0 else None,
                )
                final_output = gens[0].strip()

                # Scoring, mirroring graph_coordinate (identical metric fields).
                # Latent agents exchange no text -> communication is -1 (as graph).
                self.evaluator.metrics["communication_score"].append(-1)
                try:
                    self.evaluator.evaluate_planning(
                        final_output, agent_profiles, agent_tasks_str, final_output
                    )
                    self.evaluator.evaluate_kpi(self.task, final_output)
                except Exception:
                    self.logger.exception("graph-latent planning/KPI evaluation failed.")
                    self.evaluator.metrics["planning_score"].append(-1)

                # Stop decision: decoded true/false (replaces decide_next_step).
                solved = model.decode_bool(
                    past_kv,
                    "Considering the task and the latest answer, is the task fully "
                    f"and correctly solved?\nTask: {self.task[:800]}\n"
                    f"Latest answer: {final_output[:800]}",
                )
                summary_data["iterations"].append(
                    {
                        "iteration": rnd + 1,
                        "task_assignments": {a.agent_id: self.task for a in agents},
                        "task_results": [{judger.agent_id: final_output}],
                        "summary": final_output,
                        "continue_simulation": not solved,
                        "communications": [],
                        "total_milestones": 0,
                        "agent_kpis": {},
                    }
                )
                self.logger.info(
                    f"[graph-latent] round {rnd + 1} done (solved={solved})."
                )
                if solved:
                    break

            # Top-level metrics: identical fields to graph_coordinate.
            summary_data["planning_scores"] = self.evaluator.metrics["planning_score"]
            summary_data["communication_scores"] = self.evaluator.metrics[
                "communication_score"
            ]
            summary_data["agent_kpis"] = self.evaluator.metrics["agent_kpis"]
            summary_data["total_milestones"] = self.evaluator.metrics["total_milestones"]
            summary_data["token_usage"] = self._get_totoal_token_usage()
            # Per-environment task score (same dispatch as graph_coordinate).
            try:
                env_name = self.environment.name
                if env_name == "Research Environment":
                    self.evaluator.evaluate_task_research(self.task, final_output)
                    summary_data["task_evaluation"] = self.evaluator.metrics[
                        "task_evaluation"
                    ]
                elif env_name == "World Simulation Environment":
                    self.evaluator.evaluate_task_world(self.task, final_output)
                    summary_data["task_evaluation"] = self.evaluator.metrics[
                        "task_evaluation"
                    ]
                elif env_name == "DB Environment":
                    self.evaluator.evaluate_task_db(
                        self.task,
                        final_output,
                        self.config.task["labels"],
                        self.config.task["number_of_labels_pred"],
                        self.config.task["root_causes"],
                    )
                    summary_data["task_evaluation"] = self.evaluator.metrics[
                        "task_evaluation"
                    ]
            except Exception:
                self.logger.exception("graph-latent task evaluation failed.")
            # Latent-only extras (added on top of the standard fields).
            summary_data["final_output"] = final_output
            summary_data["total_latent_steps"] = total_latent_steps
            summary_data["decoded_tokens"] = (
                int(model.tokenize_text(final_output).shape[-1]) if final_output else 0
            )
            # Code-generation task: persist the judger-decoded solution.
            if self.environment.name == "Coding Environment":
                self._write_latent_solution(final_output)
        except Exception:
            self.logger.exception("An error occurred during graph-latent coordination.")
            raise
        finally:
            self.evaluator.finalize()
            self._write_to_jsonl(summary_data)
            self.logger.info("Graph-latent coordination simulation completed.")

    def _write_latent_solution(self, final_output: str) -> None:
        """
        Extract a python code block from the judger output and write it to the
        coding workspace as solution.py (the artifact the coder tool would have
        produced in the text baseline).
        """
        import os
        import re

        match = re.search(r"```python(.*?)```", final_output, re.DOTALL)
        code = match.group(1).strip() if match else final_output.strip()
        workspace = getattr(self.environment, "workspace_dir", "workspace")
        os.makedirs(workspace, exist_ok=True)
        path = os.path.join(workspace, "solution.py")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            self.logger.info(f"[graph-latent] wrote solution to {path}")
        except IOError as e:
            self.logger.error(f"Failed to write latent solution: {e}")

    def start(self) -> None:
        """
        Start the engine to run the simulation.
        """
        self.logger.info("Engine starting simulation.")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.launch()
        if self.coordinate_mode == "star":
            self.logger.info("Running in centralized coordination mode.")
            self.star_coordinate()
        elif self.coordinate_mode == "latent":
            self.logger.info("Running in latent (LatentMAS) coordination mode.")
            self.latent_coordinate()
        elif self.coordinate_mode == "fixed_chain":
            self.logger.info("Running in fixed-chain (text baseline) coordination mode.")
            self.fixed_chain_coordinate()
        elif self.coordinate_mode == "graph_latent":
            self.logger.info("Running in graph-latent coordination mode.")
            self.graph_latent_coordinate()
        elif self.coordinate_mode == "graph":
            self.logger.info("Running in graph-based coordination mode.")
            self.graph_coordinate()
        elif self.coordinate_mode == "chain":
            self.logger.info("Running in chain-based coordination mode.")
            self.chain_coordinate()
        elif self.coordinate_mode == "tree":
            self.logger.info("Running in tree-based coordination mode.")
            self.tree_coordinate()
        else:
            self.logger.error(f"Unsupported coordinate mode: {self.coordinate_mode}")
            raise ValueError(f"Unsupported coordinate mode: {self.coordinate_mode}")
        if isinstance(self.environment, MinecraftEnvironment):
            self.environment.finish()

    def _should_terminate(self) -> bool:
        """
        Determine whether the simulation should terminate.

        Returns:
            bool: True if should terminate, False otherwise.
        """
        # Placeholder for any additional termination conditions
        return False

    def _summarize_results(self, agents_results: List[Dict[str, Any]]) -> str:
        """
        Summarize the agents' results into a string.

        Args:
            agents_results (Dict[str, Any]): The results from all agents.

        Returns:
            str: The summary string.
        """
        summary = "Agents' Results Summary:\n"
        # for agent_id, result in agents_results.items():
        #     summary += f"- {agent_id}: {result}\n"
        for result in agents_results:
            shorten_result = f"- {result}"
            shorten_result = shorten_result[:1000]
            summary += f"{shorten_result}\n"

        self.logger.debug(f"Summarized agents' results:\n{summary}")
        return summary

    def _write_to_jsonl(self, summary_data: Dict[str, Any]) -> None:
        """
        Write summary data to the JSONL file.

        Args:
            summary_data (List[Dict[str, Any]]): Summary data to write to the JSONL file.
        """
        file_path = self.config.output.get(
            "file_path", "result/discussion_output.jsonl"
        )
        try:
            with open(file_path, "a") as jsonl_file:
                print(summary_data)
                jsonl_file.write(json.dumps(summary_data) + "\n")

                jsonl_file.flush()
            self.logger.info(f"Summary data successfully written to {file_path}")
        except IOError as e:
            self.logger.error(f"Failed to write summary data to {file_path}: {e}")

    def _get_final_ooutput_in_graph(self):
        """
        Get the final output graph.

        Returns:
            Dict[str, Any]: The final output graph.
        """
        return self.graph.get_output_graph()

    def _format_communications(self, communications: List[Any]) -> str:
        """
        Formats the communications list into a string suitable for evaluator input.
        """
        # Assuming each communication is a string or can be converted to string
        return "\n".join(str(c) for c in communications)

    def _get_agent_profiles(self) -> str:
        """
        Retrieves and formats agent profiles into a string.
        """
        agent_profiles = []
        for agent in self.graph.get_all_agents():
            # Assuming agent has attributes agent_id and profile
            agent_profiles.append(
                f"Agent ID: {agent.agent_id}, Profile: {agent.profile}"
            )
        return "\n".join(agent_profiles)

    def _format_agent_tasks(self, agent_tasks: Dict[str, Any]) -> str:
        """
        Formats agent tasks into a string.
        """
        try:
            return "\n".join(
                f"Agent {agent_id}: Task: {task}"
                for agent_id, task in agent_tasks.items()
            )
        except Exception:
            return "\n".join(json.dumps(item) for item in agent_tasks)

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Formats results into a string.
        """
        results_str = []
        for result in results:
            if "agent_id" in result and "result" in result:
                agent_id = result["agent_id"]
                res_content = result["result"]
                results_str.append(f"AgentID: {agent_id}: Result: {res_content}")
            else:
                for agent_id, res_content in result.items():
                    results_str.append(f"Agent {agent_id}: Result: {res_content}")
        return "\n".join(results_str)

    def _get_totoal_token_usage(self) -> int:
        """
        Get the total token usage by agents, planner, and LLM-backed tools.
        """
        return (
            sum(agent.token_usage for agent in self.graph.get_all_agents())
            + self.planner.token_usage
            + getattr(self.environment, "token_usage", 0)
        )
