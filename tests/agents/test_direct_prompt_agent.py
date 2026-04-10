"""
Test Script for DirectPromptAgent Class

This script tests the functionality of the DirectPromptAgent by:
1. Loading the OpenAI API key from environment variables
2. Instantiating the agent
3. Sending a test prompt
4. Printing the response and explaining the knowledge source

"""

from src.workflow_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")


# Define the test prompt
prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(openai_api_key)

# Send the prompt to the agent and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the prompt and response
print(f"\nPrompt: {prompt}")
print(f"\nResponse from DirectPromptAgent:")
print(direct_agent_response)

# Print an explanatory message describing the knowledge source
print("\n" + "=" * 60)
print("Knowledge Source Explanation:")
print("=" * 60)
print("The DirectPromptAgent uses the general knowledge embedded in the LLM")
print("(gpt-3.5-turbo model) during its training. It does not use any external")
print("knowledge base, system prompts, or additional context - only the knowledge")
print("that was learned during the model's pre-training phase on large text corpora.")
print("This makes it the simplest form of LLM interaction.")
