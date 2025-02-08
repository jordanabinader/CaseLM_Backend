from typing import Dict, Any, List, Annotated
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class DiscussionState(TypedDict):
    case_content: str
    current_step: str
    personas: Dict[str, Any]
    discussion_plan: Dict[str, Any]
    current_discussion: Annotated[List[Dict[str, Any]], add_messages]
    user_inputs: Annotated[List[Dict[str, Any]], add_messages]
    summaries: Annotated[List[Dict[str, Any]], add_messages]
    evaluations: Annotated[List[Dict[str, Any]], add_messages]
    complete: bool

class CaseDiscussionWorkflow:
    def __init__(self):
        from src.agents.orchestrator_agent import OrchestratorAgent
        from src.agents.planner_agent import PlannerAgent
        from src.agents.executor_agent import ExecutorAgent
        from src.agents.persona_creator_agent import PersonaCreatorAgent
        from src.agents.evaluator_agent import EvaluatorAgent
        from src.agents.summarizer_agent import SummarizerAgent

        self.orchestrator = OrchestratorAgent()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.persona_creator = PersonaCreatorAgent()
        self.evaluator = EvaluatorAgent()
        self.summarizer = SummarizerAgent()

        self.workflow = StateGraph(DiscussionState)

        self.setup_nodes()
        self.setup_edges()

        self.graph = self.workflow.compile()

    def setup_nodes(self):
        self.workflow.add_node("create_personas", self.create_personas)
        self.workflow.add_node("create_plan", self.create_plan)
        self.workflow.add_node("execute_discussion", self.execute_discussion)
        self.workflow.add_node("evaluate_discussion", self.evaluate_discussion)
        self.workflow.add_node("summarize_discussion", self.summarize_discussion)
        self.workflow.add_node("orchestrate", self.orchestrate)

    def setup_edges(self):
        self.workflow.add_edge(START, "create_personas")

        self.workflow.add_conditional_edges("create_personas", self.persona_creation_condition, {"create_plan": "create_plan", END: END})
        self.workflow.add_conditional_edges("create_plan", self.planning_condition, {"execute_discussion": "execute_discussion", END: END})

        self.workflow.add_edge("execute_discussion", "evaluate_discussion")
        self.workflow.add_edge("evaluate_discussion", "summarize_discussion")
        self.workflow.add_edge("summarize_discussion", "orchestrate")

        self.workflow.add_conditional_edges("orchestrate", self.orchestration_condition, {"execute_discussion": "execute_discussion", END: END})

    async def create_personas(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.persona_creator.process({"case_content": state["case_content"]})
        return {"personas": result["personas"]}

    async def create_plan(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.planner.process({"case_content": state["case_content"], "personas": state["personas"]})
        return {"discussion_plan": result["plan"]}

    async def execute_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.executor.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "personas": state["personas"],
            "current_discussion": state["current_discussion"]
        })
        return {"current_discussion": [result["discussion"]]}

    async def evaluate_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.evaluator.process({
            "current_discussion": state["current_discussion"],
            "user_inputs": state["user_inputs"],
            "discussion_plan": state["discussion_plan"]
        })
        return {"evaluations": [result["evaluation"]]}

    async def summarize_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.summarizer.process({
            "current_discussion": state["current_discussion"],
            "evaluations": state["evaluations"]
        })
        return {"summaries": [result["summary"]]}

    async def orchestrate(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.orchestrator.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "evaluations": state["evaluations"],
            "summaries": state["summaries"]
        })

        return {"complete": result["next_action"] == "complete", "current_step": result["next_action"]}

    def persona_creation_condition(self, state: DiscussionState) -> str:
        return "create_plan" if state["personas"] else END

    def planning_condition(self, state: DiscussionState) -> str:
        return "execute_discussion" if state["discussion_plan"] else END

    def orchestration_condition(self, state: DiscussionState) -> str:
        return END if state["complete"] else "execute_discussion"

    async def run(self, case_content: str) -> Dict[str, Any]:
        initial_state = DiscussionState(
            case_content=case_content,
            current_step="create_personas",
            personas=[],
            discussion_plan={},
            current_discussion=[],
            user_inputs=[],
            summaries=[],
            evaluations=[],
            complete=False
        )

        if not hasattr(self, 'graph'):
            self.graph = self.create_graph()

        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "personas": final_state["personas"],
            "discussion_plan": final_state["discussion_plan"],
            "discussions": final_state["current_discussion"],
            "summaries": final_state["summaries"],
            "evaluations": final_state["evaluations"]
        }

    def create_graph(self):
        """Create the workflow graph with all necessary agents"""

        graph = StateGraph(DiscussionState)
        graph.add_node("create_personas", self.create_personas)
        graph.add_node("create_plan", self.create_plan)
        graph.add_node("execute_discussion", self.execute_discussion)
        graph.add_node("evaluate_discussion", self.evaluate_discussion)
        graph.add_node("summarize_discussion", self.summarize_discussion)
        graph.add_node("orchestrate", self.orchestrate)
        
        # Add edges to define the workflow
        graph.add_edge(START, "create_personas")
        graph.add_conditional_edges("create_personas", self.persona_creation_condition, {"create_plan": "create_plan", END: END})
        graph.add_conditional_edges("create_plan", self.planning_condition, {"execute_discussion": "execute_discussion", END: END})
        graph.add_edge("execute_discussion", "evaluate_discussion")
        graph.add_edge("evaluate_discussion", "summarize_discussion")
        graph.add_edge("summarize_discussion", "orchestrate")
        graph.add_conditional_edges("orchestrate", self.orchestration_condition, {"execute_discussion": "execute_discussion", END: END})
        
        return graph.compile()