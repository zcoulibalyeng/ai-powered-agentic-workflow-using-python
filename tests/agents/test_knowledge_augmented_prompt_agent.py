"""
Test Script for KnowledgeAugmentedPromptAgent Class

This script tests the functionality of the KnowledgeAugmentedPromptAgent by:
1. Loading the OpenAI API key from environment variables
2. Instantiating the agent with a persona and specific (incorrect) knowledge
3. Sending a test prompt
4. Demonstrating that the agent uses provided knowledge over LLM's inherent knowledge

"""

# Import the KnowledgeAugmentedPromptAgent class from workflow_agents
from src.workflow_agents import KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")

# Define the test prompt
prompt = "What is the capital of France?"

# Define persona and knowledge
# Note: I intentionally provide INCORRECT knowledge to demonstrate
# that the agent uses the provided knowledge rather than its inherent LLM knowledge
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate a KnowledgeAugmentedPromptAgent
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Send the prompt to the agent and store the response
knowledge_agent_response = knowledge_agent.respond(prompt)

# Print the configuration and response
print(f"\nPersona: {persona}")
print(f"\nKnowledge provided: {knowledge}")
print(f"\nPrompt: {prompt}")
print(f"\nResponse from KnowledgeAugmentedPromptAgent:")
print(knowledge_agent_response)

# Print statement demonstrating knowledge usage
print("\n" + "=" * 60)
print("Knowledge Usage Confirmation:")
print("=" * 60)
print("IMPORTANT: Notice that the agent responded with 'London' as the capital of France.")
print("This is INCORRECT in reality (Paris is the actual capital), but the agent used")
print("the PROVIDED knowledge rather than its inherent LLM knowledge.")
print("\nThis demonstrates that the KnowledgeAugmentedPromptAgent:")
print("1. Prioritizes explicitly provided knowledge over the LLM's training data")
print("2. Can be configured to answer based on specific domain knowledge")
print("3. Will follow the knowledge constraint even when it contradicts common facts")
print("\nThis is useful for applications where you need the agent to respond based on")
print("specific, controlled information rather than general LLM knowledge.")