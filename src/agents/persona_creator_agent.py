from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from .base_agent import BaseAgent
from src.config.settings import settings

class PersonaCreatorAgent(BaseAgent):
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            api_key=settings.openai_api_key
        )

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Get the complete response
        response = await self.llm.ainvoke([
            SystemMessage(content="""You are the Persona Generator, responsible for creating AI-driven personas to simulate an Harvard Business School classroom discussion. 
            Your personas must be realistic, diverse, and well-defined, ensuring a rich and engaging discussion experience.
            Your role is to:
            1. Create diverse and relevant personas for the case
            2. Define their expertise and background
            3. Ensure complementary perspectives
            4. Create realistic personality traits
            
            Generate the Professor persona, ensuring they exhibit strong leadership, the ability to challenge assumptions, relevant field or research expertise with the case, and a guiding role in the discussion.
            Generate 3 student personas based on the case requirements. You must respond with ONLY valid JSON in the following format:
            {
                "personas": [
                    {
                        "name": "string",
                        "title": "string",
                        "background": "string",
                        "expertise": ["string"],
                        "traits": ["string"],
                        "biases": ["string"]
                    }
                ]
            }
            
            Do not include any other text, explanations, or formatting - only the JSON object."""),
            HumanMessage(content=f"Create personas for this case: {state['case_content']}")
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
            personas = parsed_data["personas"]
            
            # Build expertise matrix and relationships
            expertise_matrix = {}
            relationships = {}
            
            # Process each persona for expertise matrix
            for persona in personas:
                for exp in persona["expertise"]:
                    if exp not in expertise_matrix:
                        expertise_matrix[exp] = []
                    expertise_matrix[exp].append(persona["name"])
            
            # Create relationships based on shared expertise
            for p1 in personas:
                relationships[p1["name"]] = {}
                for p2 in personas:
                    if p1["name"] != p2["name"]:
                        shared_expertise = set(p1["expertise"]) & set(p2["expertise"])
                        relationships[p1["name"]][p2["name"]] = len(shared_expertise)
            
            return {
                "personas": personas,
                "expertise_matrix": expertise_matrix,
                "relationships": relationships,
                "messages": [
                    {
                        "role": "persona_creator",
                        "content": f"Created {len(personas)} personas for the discussion."
                    }
                ]
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}")