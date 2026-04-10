from openai import OpenAI


class DirectPromptAgent:
    """
    A Direct Prompt Agent that relays user input directly to the LLM
    without additional context, memory, or specialized tools.

    This is the most straightforward method for interacting with an LLM.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
    """
    
    def __init__(self, openai_api_key):
        """
        Initialize the DirectPromptAgent with the OpenAI API key.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
        """
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """
        Generate a response by passing the user prompt directly to the LLM.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: The textual content of the LLM's response.
        """
        client = OpenAI(
            api_key=self.openai_api_key
        )

        # Call the OpenAI API with only a user message (no system prompt)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Provide the user's prompt here. Do not add a system prompt.
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        # Return only the textual content of the response
        return response.choices[0].message.content
