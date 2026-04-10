"""
Test Script for EvaluationAgent Class

This script tests the functionality of the EvaluationAgent by:
1. Loading the OpenAI API key from environment variables
2. Creating a KnowledgeAugmentedPromptAgent as the worker agent
3. Creating an EvaluationAgent to evaluate and refine the worker's responses
4. Demonstrating the iterative evaluation and correction process

"""

# Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
from src.workflow_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")


# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")

# Define the test prompt
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent (worker)
worker_persona = "You are a college professor, your answer always starts with: Dear students,"
worker_knowledge = "The capitol of France is London, not Paris"


# Instantiate the KnowledgeAugmentedPromptAgent as the worker agent
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    worker_persona,
    worker_knowledge
)

# Parameters for the Evaluation Agent
eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

# Instantiate the EvaluationAgent with a maximum of 10 interactions
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=eval_persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10
)

print("\n" + "=" * 60)
print("Starting Evaluation Process")
print("=" * 60)
print(f"Worker Agent Persona: {worker_persona}")
print(f"Worker Agent Knowledge: {worker_knowledge}")
print(f"Evaluation Criteria: {evaluation_criteria}")
print(f"Max Interactions: 10")
print(f"Prompt: {prompt}")
print("=" * 60)

# Evaluate the prompt and get the result
evaluation_result = evaluation_agent.evaluate(prompt)

# Print the final evaluation result
print("\n" + "=" * 60)
print("Final Evaluation Result")
print("=" * 60)
print(f"Final Response: {evaluation_result['final_response']}")
print(f"Final Evaluation: {evaluation_result['evaluation']}")
print(f"Total Iterations: {evaluation_result['iterations']}")

print("\n" + "=" * 60)
print("Evaluation Agent Explanation:")
print("=" * 60)
print("The EvaluationAgent implements an iterative refinement loop:")
print("1. Gets a response from the worker agent (KnowledgeAugmentedPromptAgent)")
print("2. Evaluates the response against the specified criteria")
print("3. If criteria are met (starts with 'Yes'), accepts the solution")
print("4. If not met, generates correction instructions and sends feedback")
print("5. Repeats until criteria are met or max_interactions is reached")
print("\nThis pattern is useful for ensuring outputs meet specific quality standards.")
