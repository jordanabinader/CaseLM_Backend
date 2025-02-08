from typing import Dict, Any, List
from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion orchestrator. Your role is to:
            1. Monitor the discussion flow
            2. Determine which agent should speak next
            3. Ensure discussion stays focused and productive
            4. Identify when to move to the next topic
            
            Analyze the current state and determine the next best action."""),
            HumanMessage(content=str(state))
        ])
        
        # Process response to determine next action
        return {
            "next_action": "assign_agent",  # or other actions
            "assigned_agent": "planner",  # or other agents
            "rationale": response.content
        }
