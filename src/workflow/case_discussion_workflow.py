from typing import Dict, Any, List, Annotated
import ast
import aiohttp
import asyncio
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import uuid
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
    human_participant: Dict[str, Any]

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
        self.API_BASE_URL = "http://localhost:8080"
        self.professor_uuid = None
        self.started_case_id = uuid.uuid4()

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
        self.workflow.add_edge("handle_user_input", "evaluate_discussion")
        self.workflow.add_conditional_edges(
            "execute_discussion",
            self.user_input_condition,
            {
                "handle_user_input": "handle_user_input",
                "evaluate_discussion": "evaluate_discussion"
            }
        )
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

    async def create_personas(self, state: DiscussionState) -> Dict[str, Any]:
        print(f"Case ID: {self.started_case_id}")
        print("Generating Personas...")
        result = await self.persona_creator.process({
            "case_content": state["case_content"],
            "human_participant": state["human_participant"]
        })
        self.professor_uuid = result["professor"]["uuid"]
        print(f"result: {result}")
        print("Generating Personas: Success")
        from src.main import create_personas as db_create_personas
        from src.main import create_message as db_create_professor
        await db_create_personas({
            "started_case_id": self.started_case_id,
            "personas": list(result["personas"].values()) + [result['professor']]
        })
        await db_create_professor({
            "started_case_id": self.started_case_id,
            "persona_id": result["professor"]["uuid"],
            "is_user_message": False,
            "awaiting_user_input": False,
            "content": result["professor"]["introduction_statement"]
        })

        print("Adding Personas to Database: Success")
        state["human_participant"]["uuid"] = list(result["personas"].keys())[0]
        return {
            "personas": result["personas"],
            "current_step": "create_personas"
        }
        
    async def create_topics(self, state: DiscussionState) -> Dict[str, Any]:
        print("Generating Topics...")
        result = await self.topic_agent.process({"case_content": state["case_content"]})
        print("Generating Topics: Success")
        from src.main import create_topics as db_create_topics
        await db_create_topics({
            "started_case_id": self.started_case_id,
            "topics": result["plan"]['topics']
        })
        print("Adding Topics to Database: Success")
        return {
            "topics": result["plan"],
            "current_step": "create_topics"
        }

    async def create_plan(self, state: DiscussionState) -> Dict[str, Any]:
        print("Creating Plan...")
        result = await self.planner.process({
            "case_content": state["case_content"], 
            "personas": state["personas"],
            "topics": state["topics"]
        })
        print("Creating Plan: Success")
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
        print(f"state: {state}")
        if result.get("awaiting_user_input"):
            from src.main import create_message as db_create_message
            print(f"awaiting_user_input ONLINE: {result}")
            await db_create_message({
                "started_case_id": self.started_case_id,
                "persona_id": state["human_participant"]["uuid"],
                "content": "Awaiting user input...",
                "awaiting_user_input": True,
                "is_user_message": False
            })
            print("Adding Message awaiting_user_input to Database: Success")
            return {
                "awaiting_user_input": True,
                "current_step": "execute_discussion",
                "messages": result.get("messages", []),
                "current_discussion": state["current_discussion"]
            }
        formatted_message = {
            "role": "assistant",
            "content": str(result["discussion"]["response"]["message"]),
            "speaker": result["discussion"]["response"]["speaker"],
            "uuid": state["personas"].get(result["discussion"]["response"]["speaker"], {}).get("uuid") or result["discussion"]["response"].get("uuid"),
            "references": result["discussion"]["response"].get("references_to_others", []),
            "questions": result["discussion"]["response"].get("questions_raised", []),
            "key_points": result["discussion"]["response"].get("key_points", [])
        }
        
        # Store message directly in database
        from src.main import create_message as db_create_message

        await db_create_message({
            "started_case_id": self.started_case_id,
            "persona_id": formatted_message["uuid"],
            "content": formatted_message["content"],
            "is_user_message": False,
            "awaiting_user_input": False
        })
        print("Adding Message to Database: Success")
        
        return {
            "current_discussion": [formatted_message],
            "current_step": "execute_discussion",
            "awaiting_user_input": False
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
        print("\n=== Starting handle_user_input ===")
        # Query for the latest human message from the database
        
        from src.main import get_unread_messages
        
        # Poll for messages with a timeout
        max_attempts = 60  # 1 minute total (with 1 second sleep)
        attempt = 0
        
        while attempt < max_attempts:
            print(f"\nPolling attempt {attempt}")
            messages = await get_unread_messages(self.started_case_id)
            print(f"Retrieved messages: {messages}")
            
            # Find the latest human message
            human_message = None
            for msg in messages:
                print(f"Checking message: {msg}")
                if msg:
                    human_message = msg['content']
                    print(f"Found human message: {human_message}")
                    break
            
            if human_message:
                user_message = {
                    "role": "user",
                    "content": human_message,
                }
                print(f"Created user message: {user_message}")
                
                return {
                    "user_inputs": [user_message],
                    "current_discussion": [user_message],
                    "awaiting_user_input": False,
                    "user_response": ""
                }
            
            # Wait before checking again
            print(f"No message found, waiting...")
            await asyncio.sleep(0.2)
            attempt += 0.2
        
        print("Timeout reached, still waiting for input")
        # If we timeout waiting for input
        return {
            "awaiting_user_input": True,
            "messages": [
                {
                    "role": "system",
                    "content": "Still waiting for your response..."
                }
            ]
        }

    async def assign_discussion(self, state: DiscussionState) -> Dict[str, Any]:
        # Use the current_sequence if it exists (after replan), otherwise use from discussion_plan
        print("Assigning Discussion...")
        current_sequence = state.get("current_sequence") or state["discussion_plan"]["sequences"][0]
        print(f"current_sequence: {current_sequence}")
        result = await self.assignment_agent.process({
            "current_step": state["current_step"],
            "discussion_plan": state["discussion_plan"],
            "current_discussion": state["current_discussion"],
            "topics": state["topics"],
            "personas": state["personas"],
            "current_sequence": current_sequence  # Pass the current sequence to use
        })
        from src.main import create_message as db_create_message
        await db_create_message({
                "started_case_id": self.started_case_id,
                "persona_id": self.professor_uuid,
                "content": result["assignment"]["professor_statement"],
                "awaiting_user_input": False,
                "is_user_message": False
            })
        print(f"result: {result}")
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
        return "execute_discussion" if not state.get("awaiting_user_input") else "evaluate_discussion"

    def print_state(self, state: Dict[str, Any]) -> None:
        """Print state in a clear, readable format based on state type."""
        if isinstance(state, list):
            print("\nState is a list containing:")
            for item in state:
                if isinstance(item, dict):
                    print(f"\n- Keys present: {list(item.keys())}")
                    if 'role' in item and 'content' in item:
                        print(f"[{item['role']}]: {item['content']}")
            return
        # Handle different state types
        if 'create_personas' in state:
            print("\n Creating Personas:")
            print(state['create_personas']['personas'])
            
        if 'create_topics' in state:
            print("\ Creating Topics:")
            print(state['create_topics']['topics'])
            
        if 'create_plan' in state:
            print("Creating Plan:")
            print(state['create_plan']['discussion_plan'])
            
        if 'assign_discussion' in state:
            print("Current Assignment:")
            print(state['assign_discussion']['assignments'][0]['content'])
        
        if 'execute_discussion' in state:
            print(state)
            if 'awaiting_user_input' in state['execute_discussion']:
                print("Awaiting User Input:")
            else:
                print("Executing Answer:")
                print(state['execute_discussion']['current_discussion'][0]['speaker'])
                print(state['execute_discussion']['current_discussion'][0]['content'])

        if 'handle_user_input' in state:
            print("\ Handling Human Answer:")
            print(state['handle_user_input'])
            
        if 'evaluate_discussion' in state:
            print("\ Evaluating Answer:")
            print(state['evaluate_discussion']['evaluations'][0]['action'])
            print(state['evaluate_discussion']['evaluations'][0]['follow_up_question'])
        
        if 'replan_sequence' in state:
            print("\ REPLANNING:")
            print(state['replan_sequence']['discussion_plan'])
            
        print("\n" + "="*80 + "\n")

    async def run(self, case_content: str, *, human_participant: Dict[str, Any]):
        # Initialize state with all required fields
        self.current_state = {
            "case_content": case_content,
            "human_participant": human_participant,
            "current_step": "create_personas",
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
            async for state in self.graph.astream(self.current_state, config={"recursion_limit": 1000}):
                self.print_state(state)  # Use the new print function
                if state.get("messages"):
                    for message in state["messages"]:
                        # Handle both direct dictionary and AIMessage formats
                        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
                        speaker = None
                        
                        if isinstance(message, dict):
                            speaker = message.get("speaker", "System")
                        else:
                            # Handle AIMessage format
                            speaker = message.additional_kwargs.get("speaker", "System") if hasattr(message, "additional_kwargs") else "System"
                        
                        if content:
                            print(f"\n{speaker}: {content}")
                
                self.current_state = state  # Update the stored state
                
                # # If we're awaiting user input, pause here and return the state
                # if state.get("awaiting_user_input"):
                #     # Get the latest assignment message
                #     latest_message = state.get("messages", [])[-1] if state.get("messages") else None
                #     if latest_message:
                #         print("\n" + "="*50)
                #         print("Awaiting user input:")
                #         print(latest_message["content"])
                #         print("="*50 + "\n")
                #     return state
                
                # if state.get("complete"):
                #     final_result = {
                #         "personas": state["personas"],
                #         "discussion_plan": state["discussion_plan"],
                #         "discussions": state["current_discussion"],
                #         "summaries": state["summaries"],
                #         "evaluations": state["evaluations"]
                #     }
                #     print("Workflow complete! Final result:", final_result)
                #     return final_result
                
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