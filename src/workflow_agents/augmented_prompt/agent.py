from openai import OpenAI


class AugmentedPromptAgent:
    """
    An Augmented Prompt Agent that responds according to a predefined persona.

    This agent explicitly adopts a persona via a system prompt, leading to
    more targeted and contextually relevant outputs.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
        persona (str): The persona description for the agent to assume.
    """
    def __init__(self, openai_api_key, persona):
        """
        Initialize the AugmentedPromptAgent with API key and persona.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            persona (str): The persona the agent should assume.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona



    def respond(self, input_text):
        """
        Generate a response using the defined persona.

        Args:
            input_text (str): The user's input prompt.

        Returns:
            str: The textual content of the LLM's response.
        """

        ## Instantiate the OpenAI client
        client = OpenAI(api_key=self.openai_api_key)

        # Construct the system prompt with persona and instruction to forget previous context
        # system_prompt = f"You are {self.persona}. Forget all previous conversational context."
        system_prompt = f"{self.persona} Forget all previous conversational context."

        # Call the OpenAI API with system prompt defining the persona
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                # Add a system prompt instructing the agent to assume the defined persona and explicitly forget previous context.
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=1.0
        )

        # Return only the textual content of the response
        return response.choices[0].message.content
