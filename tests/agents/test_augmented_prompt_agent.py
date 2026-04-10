"""
Test Script for AugmentedPromptAgent Class

This script tests the functionality of the AugmentedPromptAgent by:
1. Loading the OpenAI API key from environment variables
2. Instantiating the agent with a specific persona
3. Sending a test prompt
4. Printing the response and explaining how persona affects complete-workflow-output

"""
# Import the AugmentedPromptAgent class
from src.workflow_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")

# Define the test prompt and persona
prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate the AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the prompt to the agent and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the prompt and response
print(f"\nPersona: {persona}")
print(f"\nPrompt: {prompt}")
print(f"\nResponse from AugmentedPromptAgent:")
print(augmented_agent_response)


# Explanation of knowledge source and persona impact
print("\n" + "=" * 60)
print("Knowledge Source and Persona Impact Explanation:")
print("=" * 60)

print("Knowledge Source:")
print("- The agent uses general knowledge from the gpt-3.5-turbo model's training data.")
print("- This includes factual information learned during pre-training on large corpora.")
print("- No external knowledge base or RAG system is used.")

print("\nPersona Impact:")
print("- The system prompt instructs the model to assume the 'college professor' persona.")
print("- This affects the STYLE and FORMAT of the response (starting with 'Dear students,').")
print("- The factual CONTENT remains unchanged (Paris is still the capital of France).")
print("- This demonstrates how personas shape response presentation without altering facts.")
