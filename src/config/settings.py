from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
import os
from pathlib import Path

# Get the project root directory (where .env should be located)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from the root directory
env_path = ROOT_DIR / ".env"
print(f"Environment path: {env_path}")
if not env_path.exists():
    raise FileNotFoundError(f".env file not found at {env_path}")

load_dotenv(env_path)
print(f"Environment loaded from: {env_path}")

# Debug: Print environment variables (remove in production)
print(f"OpenAI API Key present: {'OPENAI_API_KEY' in os.environ}")
print(f"OpenAI Model present: {'OPENAI_MODEL' in os.environ}")

class Settings(BaseModel):
    """Application settings."""
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API Key"
    )
    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
        description="OpenAI Model Name"
    )

    # Print API key info after initialization
    def __init__(self, **data):
        super().__init__(**data)
        # Safely print partial key for debugging
        if self.openai_api_key:
            key_start = self.openai_api_key[:4]
            key_end = self.openai_api_key[-4:]
            print(f"OpenAI API Key loaded: {key_start}...{key_end}")
        print(f"OpenAI Model: {self.openai_model}")

    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "OpenAI API key not found in environment variables. "
                f"Please ensure your .env file exists at {env_path} "
                "and contains OPENAI_API_KEY=sk-..."
            )
        if not (v.startswith("sk-") or v.startswith("sk-proj-")):
            raise ValueError("Invalid OpenAI API key format - should start with 'sk-' or 'sk-proj-'")
        return v

    @field_validator('openai_model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        valid_models = [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo",
            # Add other valid models as needed
        ]
        if v not in valid_models:
            raise ValueError(f"Invalid model name. Must be one of: {', '.join(valid_models)}")
        return v

# Create settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Error initializing settings: {e}")
    raise

if __name__ == "__main__":
    print(settings)
    