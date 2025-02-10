from typing import Dict, Any, List, Annotated
import ast
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class DiscussionState(TypedDict):
    case_content: str
    current_step: str
    personas: Dict[str, Any]
    discussion_plan: Dict[str, Any]
    topics: Dict[str, Any]
    current_discussion: Annotated[List[Dict[str, Any]], add_messages]
    user_inputs: Annotated[List[Dict[str, Any]], add_messages]
    summaries: Annotated[List[Dict[str, Any]], add_messages]
    evaluations: Annotated[List[Dict[str, Any]], add_messages]
    assignments: Annotated[List[Dict[str, Any]], add_messages]
    complete: bool
    awaiting_user_input: bool
    user_response: str

class CaseDiscussionWorkflow:
    def __init__(self):
        from src.agents.orchestrator_agent import OrchestratorAgent
        from src.agents.planner_agent import PlannerAgent
        from src.agents.executor_agent import ExecutorAgent
        from src.agents.persona_creator_agent import PersonaCreatorAgent
        from src.agents.evaluator_agent import EvaluatorAgent
        from src.agents.summarizer_agent import SummarizerAgent
        from src.agents.topic_agent import TopicAgent
        from src.agents.assignment_agent import AssignmentAgent
        from src.agents.replan_agent import ReplanAgent

        self.orchestrator = OrchestratorAgent()
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.persona_creator = PersonaCreatorAgent()
        self.evaluator = EvaluatorAgent()
        self.summarizer = SummarizerAgent()
        self.topic_agent = TopicAgent()
        self.assignment_agent = AssignmentAgent()
        self.replan_agent = ReplanAgent()

        self.workflow = StateGraph(DiscussionState)

        self.setup_nodes()
        self.setup_edges()

        self.graph = self.workflow.compile()

    def setup_nodes(self):
        self.workflow.add_node("create_personas", self.create_personas)
        self.workflow.add_node("create_topics", self.create_topics)
        self.workflow.add_node("create_plan", self.create_plan)
        self.workflow.add_node("execute_discussion", self.execute_discussion)
        self.workflow.add_node("evaluate_discussion", self.evaluate_discussion)
        self.workflow.add_node("summarize_discussion", self.summarize_discussion)
        self.workflow.add_node("orchestrate", self.orchestrate)
        self.workflow.add_node("handle_user_input", self.handle_user_input)
        self.workflow.add_node("assign_discussion", self.assign_discussion)
        self.workflow.add_node("replan_sequence", self.replan_sequence)

    def setup_edges(self):
        self.workflow.add_edge(START, "create_personas")

        self.workflow.add_conditional_edges("create_personas", self.persona_creation_condition, {"create_topics": "create_topics", END: END})
        self.workflow.add_conditional_edges("create_topics", self.topic_creation_condition, {"create_plan": "create_plan", END: END})
        self.workflow.add_conditional_edges("create_plan", self.planning_condition, {"assign_discussion": "assign_discussion", END: END})
        self.workflow.add_conditional_edges("assign_discussion", self.assignment_condition, {"execute_discussion": "execute_discussion", END: END})
        self.workflow.add_edge("execute_discussion", "evaluate_discussion")
        
        self.workflow.add_conditional_edges(
            "evaluate_discussion",
            self.evaluation_condition,
            {
                "summarize_discussion": "summarize_discussion",
                "assign_discussion": "assign_discussion",
                "replan_sequence": "replan_sequence"
            }
        )
        
        self.workflow.add_edge("summarize_discussion", "assign_discussion")
        self.workflow.add_edge("replan_sequence", "assign_discussion")

        self.workflow.add_conditional_edges(
            "execute_discussion",
            self.user_input_condition,
            {
                "handle_user_input": "handle_user_input",
                "evaluate_discussion": "evaluate_discussion"
            }
        )
        self.workflow.add_conditional_edges(
            "handle_user_input",
            self.user_input_complete_condition,
            {
                "execute_discussion": "execute_discussion",
                "evaluate_discussion": "evaluate_discussion"
            }
        )

    async def create_personas(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.persona_creator.process({"case_content": state["case_content"]})
        return {
            "personas": result["personas"],
            "current_step": "create_personas"
        }
        
    async def create_topics(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.topic_agent.process({"case_content": state["case_content"]})
        return {
            "topics": result["plan"],
            "current_step": "create_topics"
        }

    async def create_plan(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.planner.process({
            "case_content": state["case_content"], 
            "personas": state["personas"],
            "topics": state["topics"]
        })
        return {
            "discussion_plan": result["plan"],
            "current_step": "create_plan"
        }

    async def execute_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.executor.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "personas": state["personas"],
            "current_discussion": state["current_discussion"],
            "assignments": state["assignments"]
        })
        
        formatted_message = {
            "role": "assistant",
            "content": str(result["discussion"]["response"]["message"]),
            "speaker": result["discussion"]["response"]["speaker"],
            "references": result["discussion"]["response"].get("references_to_others", []),
            "questions": result["discussion"]["response"].get("questions_raised", []),
            "key_points": result["discussion"]["response"].get("key_points", [])
        }
        
        return {
            "current_discussion": [formatted_message],
            "current_step": "execute_discussion"
        }

    async def evaluate_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.evaluator.process({
            "current_discussion": state["current_discussion"],
            "user_inputs": state["user_inputs"],
            "discussion_plan": state["discussion_plan"],
            "personas": state["personas"]
        })
        
        evaluation_message = {
            "role": "assistant",
            "content": str(result["evaluation"]),
            "action": result["evaluation"].get("action", ""),
            "follow_up_question": result["evaluation"].get("follow_up_question", []),
            "suggested_next_speaker": result["evaluation"].get("suggested_next_speaker", ""),
            "sequence_complete": result["evaluation"].get("sequence_complete", False),
            "current_topic_complete": result["evaluation"].get("current_topic_complete", False)
        }
        
        # Return only the latest evaluation
        return {
            "evaluations": [evaluation_message],  # Override previous evaluations instead of appending
            "needs_replan": result["evaluation"]["action"] == "REPLAN",
            "next_topic": result["evaluation"]["action"] == "NEXT_TOPIC",
            "continue_sequence": result["evaluation"]["action"] == "CONTINUE",
            "sequence_complete": result["evaluation"].get("sequence_complete", False),
            "current_topic_complete": result["evaluation"].get("current_topic_complete", False)
        }

    async def summarize_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.summarizer.process({
            "current_discussion": state["current_discussion"],
            "evaluations": state["evaluations"]
        })
        return {
            "summaries": [
                {
                    "role": "assistant",
                    "content": str(result["summary"])
                }
            ]
        }

    async def orchestrate(self, state: DiscussionState) -> Dict[str, Any]:
        result = await self.orchestrator.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "evaluations": state["evaluations"],
            "summaries": state["summaries"]
        })

        return {"complete": result["next_step"] == "complete", "current_step": result["next_step"]}

    async def handle_user_input(self, state: DiscussionState) -> Dict[str, Any]:
        if not state["user_response"]:
            return {"awaiting_user_input": True}
        
        user_message = {
            "role": "user",
            "content": state["user_response"]
        }
        
        return {
            "user_inputs": [user_message],
            "awaiting_user_input": False,
            "user_response": ""
        }

    async def assign_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        # Use the current_sequence if it exists (after replan), otherwise use from discussion_plan
        current_sequence = state.get("current_sequence") or state["discussion_plan"]["sequences"][0]
        
        result = await self.assignment_agent.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "current_discussion": state["current_discussion"],
            "topics": state["topics"],
            "personas": state["personas"],
            "current_sequence": current_sequence  # Pass the current sequence to use
        })
        
        assignment_message = {
            "role": "system",
            "content": str(result["assignment"])
        }
        
        return {
            "assignments": [assignment_message]
        }

    async def replan_sequence(self, state: DiscussionState) -> Dict[str, Any]:
        # Get evaluations from the state
        evaluations = state.get("evaluations", [])
        if not evaluations:
            raise ValueError("No evaluations found in state")
        
        latest_evaluation = evaluations[-1]
        suggested_next_speaker = None
        
        # Handle AIMessage object
        if hasattr(latest_evaluation, 'additional_kwargs'):
            suggested_next_speaker = latest_evaluation.additional_kwargs.get("suggested_next_speaker")
        else:
            suggested_next_speaker = latest_evaluation.get("suggested_next_speaker")
        
        if not suggested_next_speaker:
            raise ValueError("No suggested next speaker found in evaluation")
        
        result = await self.replan_agent.process({
            "discussion_plan": state["discussion_plan"],
            "personas": state["personas"],
            "current_discussion": state["current_discussion"],
            "evaluations": evaluations,  # Pass the full evaluations list
            "suggested_next_speaker": suggested_next_speaker
        })
        
        # Return both the updated plan and the current sequence
        return {
            "discussion_plan": result["updated_plan"],  # This will update the main state's discussion_plan
            "current_sequence": result["updated_plan"]["sequences"][0],  # Pass the current sequence explicitly
            "current_step": "replan_sequence",
            "messages": result["messages"]
        }

    def persona_creation_condition(self, state: DiscussionState) -> str:
        return "create_topics" if state["personas"] else END

    def topic_creation_condition(self, state: DiscussionState) -> str:
        return "create_plan" if state.get("topics") else END

    def planning_condition(self, state: DiscussionState) -> str:
        return "assign_discussion" if state["discussion_plan"] else END

    def assignment_condition(self, state: DiscussionState) -> str:
        return "execute_discussion" if state.get("assignments") else END

    def evaluation_condition(self, state: DiscussionState) -> str:
        if "evaluate_discussion" in state:
            evaluations = state["evaluate_discussion"].get("evaluations", [])
        else:
            evaluations = state.get("evaluations", [])
            
        if not evaluations:
            return "assign_discussion"
            
        latest_evaluation = evaluations[-1]
        
        # Get action and completion status from additional_kwargs for AIMessage
        action = latest_evaluation.additional_kwargs.get("action", "")
        sequence_complete = latest_evaluation.additional_kwargs.get("sequence_complete", False)
        current_topic_complete = latest_evaluation.additional_kwargs.get("current_topic_complete", False)
        
        if current_topic_complete:
            return "summarize_discussion"
        elif sequence_complete:
            return "replan_sequence"
        elif action == "REPLAN":
            return "replan_sequence"
        else:
            return "assign_discussion"

    def user_input_condition(self, state: DiscussionState) -> str:
        return "handle_user_input" if state.get("awaiting_user_input") else "evaluate_discussion"

    def user_input_complete_condition(self, state: DiscussionState) -> str:
        return "execute_discussion" if state.get("awaiting_user_input") else "evaluate_discussion"

    async def run(self, case_content: str):
        # Initialize state with all required fields
        initial_state = {
            "case_content": case_content,
            "current_step": "create_personas",  # Set initial step
            "personas": {},
            "discussion_plan": {},
            "topics": {},
            "current_discussion": [],
            "user_inputs": [],
            "summaries": [],
            "evaluations": [],
            "assignments": [],
            "complete": False,
            "awaiting_user_input": False,
            "user_response": ""
        }

        if not hasattr(self, 'graph'):
            self.graph = self.workflow.compile()
            
        try:
            async for state in self.graph.astream(initial_state):
                print(f"State update: {state}")
                
                if state.get("complete"):
                    final_result = {
                        "personas": state["personas"],
                        "discussion_plan": state["discussion_plan"],
                        "discussions": state["current_discussion"],
                        "summaries": state["summaries"],
                        "evaluations": state["evaluations"]
                    }
                    print("Workflow complete! Final result:", final_result)
                    return final_result
        except Exception as e:
            print(f"Error in workflow: {str(e)}")
            raise

    def create_graph(self):
        """Create the workflow graph with all necessary agents"""
        graph = StateGraph(DiscussionState)
        
        graph.add_node("create_personas", self.create_personas)
        graph.add_node("create_topics", self.create_topics)
        graph.add_node("create_plan", self.create_plan)
        graph.add_node("assign_discussion", self.assign_discussion)
        graph.add_node("execute_discussion", self.execute_discussion)
        graph.add_node("evaluate_discussion", self.evaluate_discussion)
        graph.add_node("summarize_discussion", self.summarize_discussion)
        graph.add_node("orchestrate", self.orchestrate)
        graph.add_node("handle_user_input", self.handle_user_input)
        graph.add_node("replan_sequence", self.replan_sequence)
        
        # Add edges to define the workflow
        graph.add_edge(START, "create_personas")
        graph.add_conditional_edges("create_personas", self.persona_creation_condition, {"create_topics": "create_topics", END: END})
        graph.add_conditional_edges("create_topics", self.topic_creation_condition, {"create_plan": "create_plan", END: END})
        graph.add_conditional_edges("create_plan", self.planning_condition, {"assign_discussion": "assign_discussion", END: END})
        graph.add_conditional_edges("assign_discussion", self.planning_condition, {"execute_discussion": "execute_discussion", END: END})
        graph.add_edge("execute_discussion", "evaluate_discussion")
        graph.add_conditional_edges(
            "evaluate_discussion",
            self.evaluation_condition,
            {
                "summarize_discussion": "summarize_discussion",
                "assign_discussion": "assign_discussion",
                "replan_sequence": "replan_sequence"
            }
        )
        graph.add_edge("summarize_discussion", "assign_discussion")
        graph.add_edge("replan_sequence", "assign_discussion")
        graph.add_conditional_edges(
            "execute_discussion",
            self.user_input_condition,
            {
                "handle_user_input": "handle_user_input",
                "evaluate_discussion": "evaluate_discussion"
            }
        )
        graph.add_conditional_edges(
            "handle_user_input",
            self.user_input_complete_condition,
            {
                "execute_discussion": "execute_discussion",
                "evaluate_discussion": "evaluate_discussion"
            }
        )
        
        return graph.compile()