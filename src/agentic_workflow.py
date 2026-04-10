"""
Agentic Workflow for Technical Project Management

This script implements an AI-powered agentic workflow for project management.
It orchestrates multiple specialized agents to transform a product specification
into user stories, product features, and detailed engineering tasks.

The workflow demonstrates:
- Action Planning: Breaking down high-level goals into actionable steps
- Intelligent Routing: Directing tasks to appropriate specialized agents
- Knowledge-Augmented Generation: Using domain knowledge to produce outputs
- Iterative Evaluation: Ensuring outputs meet quality criteria

"""

# Import the required agents from the workflow_agents package
from src.workflow_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent
)
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI key into a variable called openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify API key is loaded
print(f"API Key loaded: {openai_api_key[:10]}..." if openai_api_key else "API Key NOT found!")

# Load the product spec document
spec_path = os.path.join(os.path.dirname(__file__), "..", "data", "specs", "Product-Spec-Email-Router.txt")
with open(spec_path, "r", encoding="utf-8") as file:
    product_spec = file.read()

print(f"Product specification loaded successfully. Length: {len(product_spec)} characters")

# ============================================================================
# INSTANTIATE ALL THE AGENTS
# ============================================================================

# ----------------------------------------------------------------------------
# Action Planning Agent
# ----------------------------------------------------------------------------
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)#

# ----------------------------------------------------------------------------
# Product Manager - Knowledge Augmented Prompt Agent
# ----------------------------------------------------------------------------
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."


# Complete this knowledge string by appending the product_spec loaded in TODO 3
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"\n\nProduct Specification:\n{product_spec}"
)

# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_product_manager,
    knowledge_product_manager
)


# ----------------------------------------------------------------------------
# Product Manager - Evaluation Agent
# ----------------------------------------------------------------------------
# Define the persona and evaluation criteria for a Product Manager evaluation agent
persona_product_manager_eval = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] so that [benefit/value]."
)

# Instantiate the Product Manager Evaluation Agent
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)

# ----------------------------------------------------------------------------
# Program Manager - Knowledge Augmented Prompt Agent
# ----------------------------------------------------------------------------
# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."

knowledge_program_manager = ("Features of a product are defined by organizing similar user stories into cohesive groups. "
    f"\n\nProduct Specification:\n{product_spec}"
)

# Instantiate the Program Manager Knowledge Agent (required before TODO 8)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_program_manager,
    knowledge_program_manager
)


# ----------------------------------------------------------------------------
# Program Manager - Evaluation Agent
# ----------------------------------------------------------------------------
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_program_manager = (
    "The answer should be product features that follow the following structure: "
    "Feature Name: A clear, concise title that identifies the capability\n"
    "Description: A brief explanation of what the feature does and its purpose\n"
    "Key Functionality: The specific capabilities or actions the feature provides\n"
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)

# ----------------------------------------------------------------------------
# Development Engineer - Knowledge Augmented Prompt Agent
# ----------------------------------------------------------------------------
# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = ( "Development tasks are defined by identifying what needs to be built to implement each user story. "
    f"\n\nProduct Specification:\n{product_spec}"
)

# Instantiate the Development Engineer Knowledge Agent (required before TODO 9)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key,
    persona_dev_engineer,
    knowledge_dev_engineer
)

# ----------------------------------------------------------------------------
# Development Engineer - Evaluation Agent
# ----------------------------------------------------------------------------
# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."

evaluation_criteria_dev_engineer = (
    "The answer should be tasks following this exact structure: "
    "Task ID: A unique identifier for tracking purposes\n"
    "Task Title: Brief description of the specific development work\n"
    "Related User Story: Reference to the parent user story\n"
    "Description: Detailed explanation of the technical work required\n"
    "Acceptance Criteria: Specific requirements that must be met for completion\n"
    "Estimated Effort: Time or complexity estimation\n"
    "Dependencies: Any tasks that must be completed first"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)


# ----------------------------------------------------------------------------
# Support Functions for Routing Agent
# ----------------------------------------------------------------------------
# Support functions for the routes of the routing agent

def product_manager_support_function(query):
    """
    Support function for the Product Manager route.

    Takes an input query, generates a response using the Product Manager
    Knowledge Agent, evaluates it, and returns the validated response.

    Args:
        query (str): The input query/step from the action plan.

    Returns:
        str: The final validated response containing user stories.
    """
    print("\n[Product Manager Support Function] Processing query...")

    # Step 1: Get response from the Knowledge Agent
    response_from_knowledge_agent = product_manager_knowledge_agent.respond(query)

    # Step 2: Evaluate the response using the Evaluation Agent
    evaluation_result = product_manager_evaluation_agent.evaluate(response_from_knowledge_agent)

    # Step 3: Return the final validated response
    return evaluation_result['final_response']


def program_manager_support_function(query):
    """
    Support function for the Program Manager route.

    Takes an input query, generates a response using the Program Manager
    Knowledge Agent, evaluates it, and returns the validated response.

    Args:
        query (str): The input query/step from the action plan.

    Returns:
        str: The final validated response containing product features.
    """
    print("\n[Program Manager Support Function] Processing query...")

    # Step 1: Get response from the Knowledge Agent
    response_from_knowledge_agent = program_manager_knowledge_agent.respond(query)

    # Step 2: Evaluate the response using the Evaluation Agent
    evaluation_result = program_manager_evaluation_agent.evaluate(response_from_knowledge_agent)

    # Step 3: Return the final validated response
    return evaluation_result['final_response']


def development_engineer_support_function(query):
    """
    Support function for the Development Engineer route.

    Takes an input query, generates a response using the Development Engineer
    Knowledge Agent, evaluates it, and returns the validated response.

    Args:
        query (str): The input query/step from the action plan.

    Returns:
        str: The final validated response containing engineering tasks.
    """
    print("\n[Development Engineer Support Function] Processing query...")

    # Step 1: Get response from the Knowledge Agent
    response_from_knowledge_agent = development_engineer_knowledge_agent.respond(query)

    # Step 2: Evaluate the response using the Evaluation Agent
    evaluation_result = development_engineer_evaluation_agent.evaluate(response_from_knowledge_agent)

    # Step 3: Return the final validated response
    return evaluation_result['final_response']


# ----------------------------------------------------------------------------
# Routing Agent
# ----------------------------------------------------------------------------
# Initialize the routing agent with an empty agents list
routing_agent = RoutingAgent(openai_api_key, [])


# Define the routes for the routing agent
routes = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories.",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Responsible for defining product features by grouping related user stories. Does not define individual stories or engineering tasks.",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for defining development tasks and engineering work required to implement user stories. Creates technical specifications and task breakdowns.",
        "func": lambda x: development_engineer_support_function(x)
    }
]

# Assign the routes to the routing agent's agents attribute
routing_agent.agents = routes


# ============================================================================
# RUN THE WORKFLOW
# ============================================================================

print("\n" + "=" * 80)
print("*** Workflow execution started ***")
print("=" * 80)

# Workflow Prompt - The high-level task for the TPM
# workflow_prompt = "What would the development tasks for this product be?"
workflow_prompt = "Create a comprehensive project plan for the Email Router product including user stories, key features, and development tasks."

print(f"\nTask to complete in this workflow:")
print(f"Workflow Prompt: {workflow_prompt}")

print("\n" + "-" * 80)
print("Step 1: Defining workflow steps from the workflow prompt using Action Planning Agent")
print("-" * 80)


# Implement the workflow

# Step 1: Use the action_planning_agent to extract steps from the workflow_prompt
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

print(f"\nExtracted {len(workflow_steps)} workflow steps:")
for i, step in enumerate(workflow_steps, 1):
    print(f"  {i}. {step}")

# Step 2: Initialize an empty list to store completed_steps
completed_steps = []

print("\n" + "-" * 80)
print("Step 2: Processing each workflow step through the Routing Agent")
print("-" * 80)

# Step 3: Loop through the extracted workflow steps
for i, step in enumerate(workflow_steps, 1):
    print(f"\n{'=' * 80}")
    print(f"Processing Step {i}/{len(workflow_steps)}: {step}")
    print("=" * 80)

    # Use the routing_agent to route the step to the appropriate support function
    result = routing_agent.route(step)

    # Append the result to completed_steps
    completed_steps.append(result)

    # Print the result of the current step
    print(f"\n--- Result for Step {i} ---")
    print(result)
    print("-" * 40)

# Step 4: Print the final complete-workflow-output of the workflow
print("\n" + "=" * 80)
print("*** WORKFLOW COMPLETED ***")
print("=" * 80)

print("\n" + "=" * 80)
print("FINAL WORKFLOW OUTPUT")
print("=" * 80)

# Print all completed steps with their results
for i, result in enumerate(completed_steps, 1):
    print(f"\n--- Completed Step {i} ---")
    print(result)
    print("-" * 40)

# The final complete-workflow-output is typically the last completed step
print("\n" + "=" * 80)
print("FINAL DELIVERABLE (Last Completed Step)")
print("=" * 80)
print(completed_steps[-1] if completed_steps else "No steps completed")

print("\n*** Workflow execution finished ***")
