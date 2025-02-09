# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.config.settings import settings

class TopicAgent:
    """Agent responsible for topic selection for the case study discussion."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create or update the discussion plan."""
        #TODO: Use pydantic to validate the response
        # Get planner's output
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Topic Planner, responsible for identifying the 3 most critical topics or core insights for this Harvard Business School case study discussion.
            Your role is to create a focused roadmap covering the 3 most important aspects that will lead to meaningful learning outcomes.
            For each of the 3 topics, you will:
            1. Identify a key aspect of the case that warrants deep discussion
            2. Develop questions that will drive meaningful debate
            3. Outline expected insights students should discover
            4. Structure the flow to build toward actionable conclusions
                          
            You must respond with ONLY valid JSON in the following format:
            {
                "plan": {
                    "topics": [
                        {
                            "title": "string",
                            "expected_insights": ["string"]
                        }
                    ],
                    "sequence": [0, 1, 2],
                    "status": "created"
                }
            }
            
            Ensure you provide exactly 3 topics in the response.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Develop a focused three-topic discussion plan. Include cold-call opportunities, debate questions, and key moments for insight evaluation. Create a discussion plan for this case: {state['case_content']}")
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
                "plan": parsed_data["plan"],
                "messages": [
                    {
                        "role": "planner",
                        "content": f"Discussion plan created with {len(parsed_data['plan']['topics'])} topics."
                    }
                ]
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")