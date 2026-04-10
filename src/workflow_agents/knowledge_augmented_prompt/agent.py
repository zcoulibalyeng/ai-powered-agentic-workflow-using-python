from openai import OpenAI


class KnowledgeAugmentedPromptAgent:
    """
    A Knowledge Augmented Prompt Agent that incorporates specific knowledge
    alongside a defined persona when responding to prompts.

    This agent ensures answers are based on explicitly provided information
    rather than the LLM's inherent knowledge.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
        persona (str): The persona description for the agent.
        knowledge (str): The specific knowledge the agent should use.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        """
        Initialize the KnowledgeAugmentedPromptAgent.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            persona (str): The persona the agent should assume.
            knowledge (str): The specific knowledge to use for responses.
        """
        self.persona = persona
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def respond(self, input_text):
        """
        Generate a response using the persona and provided knowledge only.

        Args:
            input_text (str): The user's input prompt.

        Returns:
            str: The textual content of the LLM's response.
        """

        # Instantiate the OpenAI client
        client = OpenAI(api_key=self.openai_api_key)

        # Construct the system message with persona, knowledge, and instructions
        system_prompt = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
            f"Answer the prompt based on this knowledge, not your own."
        )

        # Call the OpenAI API with the constructed messages
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        # Return only the textual content of the response
        return response.choices[0].message.content
