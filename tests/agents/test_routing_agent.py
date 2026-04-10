
"""
Test Script for RoutingAgent Class

This script tests the functionality of the RoutingAgent by:
1. Loading the OpenAI API key from environment variables
2. Creating multiple specialized KnowledgeAugmentedPromptAgents
3. Configuring the RoutingAgent to route prompts to appropriate agents
4. Testing routing with prompts that should match different agents

"""

# Import the KnowledgeAugmentedPromptAgent and RoutingAgent classes
from src.workflow_agents import KnowledgeAugmentedPromptAgent, RoutingAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")


# Define the Texas Knowledge Augmented Prompt Agent
texas_persona = "You are a college professor"
texas_knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, texas_persona, texas_knowledge)


# Define the Europe Knowledge Augmented Prompt Agent
europe_persona = "You are a college professor"
europe_knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, europe_persona, europe_knowledge)


# Define the Math Knowledge Augmented Prompt Agent
math_persona = "You are a college math professor"
math_knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, math_persona, math_knowledge)


# Instantiate the RoutingAgent with an empty agents list initially
routing_agent = RoutingAgent(openai_api_key, {})


# Define the agents list with name, description, and function for each agent
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x)
    }
]


# Assign the agents to the RoutingAgent
routing_agent.agents = agents

print("\n" + "=" * 60)
print("RoutingAgent Test - Testing Semantic Routing")
print("=" * 60)
print("Available Agents:")
for agent in agents:
    print(f"  - {agent['name']}: {agent['description']}")
print("=" * 60)

# Test prompts to demonstrate routing
test_prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

# Test each prompt and print the routing result
for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'=' * 60}")
    print(f"Test {i}: {prompt}")
    print("=" * 60)

    response = routing_agent.route(prompt)

    print(f"\nRouted Response:")
    print(response)

print("\n" + "=" * 60)
print("RoutingAgent Explanation:")
print("=" * 60)
print("The RoutingAgent uses semantic similarity to route prompts:")
print("1. Calculates embedding for the user's prompt using text-embedding-3-large")
print("2. Calculates embedding for each agent's description")
print("3. Computes cosine similarity between prompt and each description")
print("4. Routes to the agent with the highest similarity score")
print("\nThis allows for intelligent, context-aware routing without explicit rules.")
print("\nExpected Routing:")
print("  - 'Rome, Texas' -> Texas Agent (Texas-related content)")
print("  - 'Rome, Italy' -> Europe Agent (Europe-related content)")
print("  - 'story takes 2 days' -> Math Agent (contains numbers/calculations)")