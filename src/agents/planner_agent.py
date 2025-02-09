# src/agents/planner_agent.py
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
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
        #TODO: Use pydantic to validate the response
        # Get planner's output
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Planner, responsible for determining the sequence of personas in the discussion of each topic.
            Your role is to create an engaging discussion flow by ordering the personas in a way that builds meaningful dialogue and insights.
            DO NOT include the professor in the sequence. but make sure to include all other personas.
            You must respond with ONLY valid JSON in the following format:
            {
                "plan": {
                    "sequences": [
                        {
                            "topic_index": int,
                            "persona_sequence": ["persona_id1", "persona_id2", "persona_id3"]
                        }
                    ],
                    "status": "created"
                }
            }
            
            The persona_sequence should list the IDs of personas in the order they should speak.
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"""Create a discussion sequence for each topic.
                Case content: {state['case_content']}
                Topics: {state['topics']}
                Available personas: {state['personas']}""")
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
                        "content": f"Discussion sequences created for {len(parsed_data['plan']['sequences'])} topics."
                    }
                ]
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")