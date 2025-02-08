from typing import Dict, Any, List
from .base_agent import BaseAgent

class PlannerAgent(BaseAgent):
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion planner. Your role is to:
            1. Create a structured plan for the case discussion
            2. Define key discussion points and their sequence
            3. Specify which personas should be involved in each point
            4. Set clear objectives for each discussion segment
            
            Based on the case content and current state, create or update the discussion plan."""),
            HumanMessage(content=str(state))
        ])
        
        # Parse plan from response
        return {
            "plan": response.content,
            "discussion_points": [],  # Parsed discussion points
            "sequence": []  # Discussion sequence
        }
