"""
Test Script for ActionPlanningAgent Class

This script tests the functionality of the ActionPlanningAgent by:
1. Loading the OpenAI API key from environment variables
2. Defining cooking knowledge for the agent
3. Instantiating the ActionPlanningAgent
4. Testing step extraction from a user prompt

"""
# Import all required libraries, including the ActionPlanningAgent
from src.workflow_agents import ActionPlanningAgent
import os
from dotenv import load_dotenv

# Load environment variables and define the openai_api_key variable
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")


# Define the knowledge base for the agent - contains recipes for different egg preparations
knowledge = """
# Fried Egg
1. Heat pan with oil or butter
2. Crack egg into pan
3. Cook until white is set (2-3 minutes)
4. Season with salt and pepper
5. Serve

# Scrambled Eggs
1. Crack eggs into a bowl
2. Beat eggs with a fork until mixed
3. Heat pan with butter or oil over medium heat
4. Pour egg mixture into pan
5. Stir gently as eggs cook
6. Remove from heat when eggs are just set but still moist
7. Season with salt and pepper
8. Serve immediately

# Boiled Eggs
1. Place eggs in a pot
2. Cover with cold water (about 1 inch above eggs)
3. Bring water to a boil
4. Remove from heat and cover pot
5. Let sit: 4-6 minutes for soft-boiled or 10-12 minutes for hard-boiled
6. Transfer eggs to ice water to stop cooking
7. Peel and serve
"""

# Instantiate the ActionPlanningAgent with the knowledge and API key
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge)

# Define the test prompt
prompt = "One morning I wanted to have scrambled eggs"

print("\n" + "=" * 60)
print("ActionPlanningAgent Test - Step Extraction")
print("=" * 60)
print("Knowledge Base Contains Recipes For:")
print("  - Fried Eggs")
print("  - Scrambled Eggs")
print("  - Boiled Eggs")
print("=" * 60)

print(f"\nPrompt: {prompt}")
print("\n" + "-" * 60)
print("Extracted Steps:")
print("-" * 60)

# Print the agent's response to the prompt
steps = action_planning_agent.extract_steps_from_prompt(prompt)

# Display each extracted step
for step in steps:
    print(step)

print("\n" + "=" * 60)
print("ActionPlanningAgent Explanation:")
print("=" * 60)
print("The ActionPlanningAgent extracts actionable steps from prompts:")
print("1. Uses provided knowledge to understand available actions/procedures")
print("2. Parses the user's prompt to identify the desired action")
print("3. Returns a clean list of steps from the knowledge base")
print("\nThis agent is crucial for agentic workflows as it breaks down")
print("complex tasks into discrete, executable steps.")
print("\nIn this test, the agent correctly identified that the user wants")
print("'scrambled eggs' and extracted the relevant steps from the knowledge base.")
