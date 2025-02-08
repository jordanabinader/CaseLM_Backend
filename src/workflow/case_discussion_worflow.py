# src/workflow/case_discussion_workflow.py
from typing import Dict, Any, List, Annotated
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

class DiscussionState(BaseModel):
    """State for the case discussion workflow"""
    case_content: str
    current_step: str
    personas: Dict[str, Any] = {}
    discussion_plan: Dict[str, Any] = {}
    current_discussion: List[Dict[str, Any]] = []
    user_inputs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    evaluations: List[Dict[str, Any]] = []
    complete: bool = False

class CaseDiscussionWorkflow:
    """Manages the workflow for case discussions using LangGraph"""
    
    def __init__(self):
        from src.agents.orchestrator_agent import OrchestratorAgent
        from src.agents.planner_agent import PlannerAgent
        from src.agents.executor_agent import ExecutorAgent
        from src.agents.persona_creator_agent import PersonaCreatorAgent
        from src.agents.evaluator_agent import EvaluatorAgent
        from src.agents.summarizer_agent import SummarizerAgent
        
        # Initialize agents
        self.orchestrator = OrchestratorAgent()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.persona_creator = PersonaCreatorAgent()
        self.evaluator = EvaluatorAgent()
        self.summarizer = SummarizerAgent()
        
        # Create workflow
        self.workflow = StateGraph(DiscussionState)
        
        # Add nodes
        self.setup_nodes()
        
        # Add edges
        self.setup_edges()
        
        # Compile graph
        self.graph = self.workflow.compile()
    
    def setup_nodes(self):
        """Set up nodes in the workflow graph"""
        
        # Persona Creation Node
        self.workflow.add_node("create_personas", self.create_personas)
        
        # Planning Node
        self.workflow.add_node("create_plan", self.create_plan)
        
        # Execution Node
        self.workflow.add_node("execute_discussion", self.execute_discussion)
        
        # Evaluation Node
        self.workflow.add_node("evaluate_discussion", self.evaluate_discussion)
        
        # Summary Node
        self.workflow.add_node("summarize_discussion", self.summarize_discussion)
        
        # Orchestration Node
        self.workflow.add_node("orchestrate", self.orchestrate)
    
    def setup_edges(self):
        """Set up edges in the workflow graph"""
        
        # Add conditional edges
        self.workflow.add_edge_condition(
            "create_personas",
            self.persona_creation_condition
        )
        
        self.workflow.add_edge_condition(
            "create_plan",
            self.planning_condition
        )
        
        self.workflow.add_edge_condition(
            "execute_discussion",
            self.execution_condition
        )
        
        self.workflow.add_edge_condition(
            "evaluate_discussion",
            self.evaluation_condition
        )
        
        self.workflow.add_edge_condition(
            "summarize_discussion",
            self.summary_condition
        )
        
        self.workflow.add_edge_condition(
            "orchestrate",
            self.orchestration_condition
        )
    
    async def create_personas(self, state: DiscussionState) -> DiscussionState:
        """Create personas for the case discussion"""
        result = await self.persona_creator.process({
            "case_content": state.case_content
        })
        state.personas = result["personas"]
        return state
    
    async def create_plan(self, state: DiscussionState) -> DiscussionState:
        """Create discussion plan"""
        result = await self.planner.process({
            "case_content": state.case_content,
            "personas": state.personas
        })
        state.discussion_plan = result["plan"]
        return state
    
    async def execute_discussion(self, state: DiscussionState) -> DiscussionState:
        """Execute current discussion step"""
        result = await self.executor.process({
            "current_step": state.current_step,
            "discussion_plan": state.discussion_plan,
            "personas": state.personas,
            "current_discussion": state.current_discussion
        })
        state.current_discussion.append(result["discussion"])
        return state
    
    async def evaluate_discussion(self, state: DiscussionState) -> DiscussionState:
        """Evaluate discussion and user input"""
        result = await self.evaluator.process({
            "current_discussion": state.current_discussion,
            "user_inputs": state.user_inputs,
            "discussion_plan": state.discussion_plan
        })
        state.evaluations.append(result["evaluation"])
        return state
    
    async def summarize_discussion(self, state: DiscussionState) -> DiscussionState:
        """Summarize current discussion state"""
        result = await self.summarizer.process({
            "current_discussion": state.current_discussion,
            "evaluations": state.evaluations
        })
        state.summaries.append(result["summary"])
        return state
    
    async def orchestrate(self, state: DiscussionState) -> DiscussionState:
        """Orchestrate the workflow"""
        result = await self.orchestrator.process({
            "current_step": state.current_step,
            "discussion_plan": state.discussion_plan,
            "evaluations": state.evaluations,
            "summaries": state.summaries
        })
        
        # Update state based on orchestrator's decision
        if result["next_action"] == "complete":
            state.complete = True
        else:
            state.current_step = result["next_action"]
        
        return state
    
    def persona_creation_condition(self, state: DiscussionState) -> str:
        """Determine next step after persona creation"""
        return "create_plan" if state.personas else END
    
    def planning_condition(self, state: DiscussionState) -> str:
        """Determine next step after planning"""
        return "execute_discussion" if state.discussion_plan else END
    
    def execution_condition(self, state: DiscussionState) -> str:
        """Determine next step after execution"""
        return "evaluate_discussion"
    
    def evaluation_condition(self, state: DiscussionState) -> str:
        """Determine next step after evaluation"""
        return "summarize_discussion"
    
    def summary_condition(self, state: DiscussionState) -> str:
        """Determine next step after summary"""
        return "orchestrate"
    
    def orchestration_condition(self, state: DiscussionState) -> str:
        """Determine next step after orchestration"""
        if state.complete:
            return END
        return "execute_discussion"
    
    async def run(self, case_content: str) -> Dict[str, Any]:
        """Run the workflow for a given case"""
        initial_state = DiscussionState(
            case_content=case_content,
            current_step="create_personas"
        )
        
        final_state = await self.graph.arun(initial_state)
        
        return {
            "personas": final_state.personas,
            "discussion_plan": final_state.discussion_plan,
            "discussions": final_state.current_discussion,
            "summaries": final_state.summaries,
            "evaluations": final_state.evaluations
        }