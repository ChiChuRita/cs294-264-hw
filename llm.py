from abc import ABC, abstractmethod
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLM(ABC):
    """Abstract base class for Large Language Models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the LLM given a prompt.
        Must include any required stop-token logic at the caller level.
        """
        raise NotImplementedError


class OpenAIModel(LLM):
    """
    Example LLM implementation using OpenAI's Responses API.

    Implements the class to call OpenAI's backend (e.g., OpenAI GPT-4 mini)
    and return the model's text output. Ensures the model produces the response
    format required by ResponseParser and includes the stop token in the output string.
    """

    def __init__(self, stop_token: str, model_name: str = "gpt-4o-mini"):
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.stop_token = stop_token
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        # Call the model and obtain text, ensuring the stop token is present
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            
            generated_text = response.choices[0].message.content
            
            # Ensure the stop token is present at the end if not already there
            if not generated_text.endswith(self.stop_token):
                generated_text += self.stop_token
                
            return generated_text
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate response from OpenAI: {str(e)}")