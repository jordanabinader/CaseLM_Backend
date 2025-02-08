from typing import Dict, Any, List
from .base_agent import BaseAgent

class EvaluatorAgent(BaseAgent):
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion evaluator. Your role is to:
            1. Analyze user input against discussion objectives
            2. Identify gaps in understanding
            3. Evaluate the quality of insights
            4. Recommend areas for deeper exploration
            
            Evaluate the current discussion state and user input."""),
            HumanMessage(content=str(state))
        ])
        
        return {
            "evaluation": response.content,
            "gaps": [],  # Identified gaps
            "recommendations": []  # Recommendations for next steps
        }
