from openai import OpenAI


class ActionPlanningAgent:
    """
    An Action Planning Agent that extracts and lists actionable steps
    from a user's prompt based on provided knowledge.

    This agent is crucial for constructing agentic workflows by breaking
    down complex tasks into discrete steps.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
        knowledge (str): The knowledge base for extracting action steps.
    """

    def __init__(self, openai_api_key, knowledge):
        """
        Initialize the ActionPlanningAgent.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            knowledge (str): The knowledge to use for action planning.
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge


    def extract_steps_from_prompt(self, prompt):
        """
        Extract actionable steps from a user prompt using provided knowledge.

        Args:
            prompt (str): The user's input prompt describing a task.

        Returns:
            list: A clean list of action steps.
        """

        # Instantiate the OpenAI client
        client = OpenAI(
            api_key=self.openai_api_key
        )

        # Define the system prompt for action planning
        system_prompt = (
            f"You are an action planning agent. Using your knowledge, you extract from the user prompt "
            f"the steps requested to complete the action the user is asking for. You return the steps as a list. "
            f"Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        # Call the OpenAI API to get the response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        # Extract the response text
        response_text = response.choices[0].message.content

        # Clean and format the extracted steps by removing empty lines
        steps = response_text.split("\n")

        # Filter out empty lines and clean up whitespace
        cleaned_steps = [step.strip() for step in steps if step.strip()]

        return cleaned_steps
