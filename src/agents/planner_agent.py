# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.prompts.agent_prompts import PLANNER_PROMPT
from src.config.settings import settings

class PlannerAgent:
    """Agent responsible for planning the case study discussion."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update the discussion plan."""
        
        # Get planner's output
        response = await self.llm.ainvoke([
            SystemMessage(content=PLANNER_PROMPT.format(
                case_content=state.get("case_content", "")
            )),
            HumanMessage(content="Please create a detailed discussion plan.")
        ])
        
        # Parse the plan from the response
        plan = self._parse_plan(response.content)
        
        return {
            "discussion_plan": plan,
            "messages": [
                {
                    "role": "planner",
                    "content": f"Discussion plan created with {len(plan['topics'])} topics."
                }
            ]
        }
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured plan."""
        # Simple parsing for now - could be made more sophisticated
        topics = []
        current_topic = ""
        
        for line in response.split('\n'):
            if line.strip():
                if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    if current_topic:
                        topics.append(current_topic)
                    current_topic = line
                else:
                    current_topic += "\n" + line
        
        if current_topic:
            topics.append(current_topic)
        
        return {
            "topics": topics,
            "sequence": list(range(len(topics))),
            "status": "created"
        }