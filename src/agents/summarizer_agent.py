from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from src.config.settings import settings

class SummarizerAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the discussion summarizer. Your role is to:
            1. Synthesize key points from the discussion
            2. Highlight important insights
            3. Track evolving perspectives
            4. Maintain context for next steps
            
            You must respond with ONLY valid JSON in the following format:
            {
                "summary": {
                    "key_points": ["string"],
                    "insights": ["string"],
                    "evolving_perspectives": ["string"],
                    "next_steps": ["string"],
                    "overall_summary": "string"
                }
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=str(state))
        ])
        
        # Parse JSON response with error handling
        import json
        try:
            # Strip any potential whitespace or markdown formatting
            cleaned_content = response.content.strip()
            if cleaned_content.startswith("```json"):
                cleaned_content = cleaned_content.split("```json")[1]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content.rsplit("```", 1)[0]
            cleaned_content = cleaned_content.strip()
            
            parsed_data = json.loads(cleaned_content)
            
            return {
                "summary": parsed_data["summary"],
                "messages": [
                    {
                        "role": "summarizer",
                        "content": f"Summary created with {len(parsed_data['summary']['key_points'])} key points."
                    }
                ]
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")