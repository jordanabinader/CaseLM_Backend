# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings
from src.models.discussion_models import DirectHumanResponse
from src.agents.base_agent import BaseAgent

class DirectHumanResponseAgent(BaseAgent):
    """Agent responsible for directly responding to the user."""
    
    def __init__(self):
        super().__init__()  
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the next part of the discussion."""
        case_content = state.get("case_content", "")
        if not case_content:
            raise ValueError("No case content provided")

        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the professor of this Harvard Business School case study.
            You are currently in the middle of a discussion with the students and you need to respond to the latest user input while awaiting more information.
            An example of what could say is: "Hmm... I see, that's super interesting! I think that...."
            You must respond with ONLY valid JSON in the following format:
            {
                "answer": {
                    "content": "string",
                    "status": "created"
                }
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Develop a quick and MASSIVELY casual and humamn response to the latest user input and case content: {state['user_inputs']}, {case_content}")
        ])
        
        try:
            print(f"response: {response}")
            parsed_data = self._clean_and_parse_response(response.content, DirectHumanResponse)
            print(f"parsed_data: {parsed_data}")
            return {
                "answer": parsed_data.model_dump()
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {e}")