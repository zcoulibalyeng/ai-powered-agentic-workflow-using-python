from openai import OpenAI
import numpy as np


class RoutingAgent():
    """
    A Routing Agent that directs user prompts to the most appropriate
    specialized agent based on semantic similarity.

    This agent calculates embeddings for prompts and agent descriptions,
    then routes to the agent with the highest similarity score.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
        agents (list): List of agent dictionaries with name, description, and func.
    """

    def __init__(self, openai_api_key, agents):
        """
        Initialize the RoutingAgent.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            agents (list): List of agent dictionaries.
        """

        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        """
        Calculate the embedding of text using the text-embedding-3-large model.

        Args:
            text (str): The text to embed.

        Returns:
            list: The embedding vector.
        """
        client = OpenAI(api_key=self.openai_api_key)

        ## Call the embeddings API with the text-embedding-3-large model
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )

        # Extract and return the embedding vector
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        """
        Route user prompts to the most appropriate agent based on similarity.

        Args:
            user_input (str): The user's input prompt.

        Returns:
            The response from the selected agent's function.
        """

        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        # Iterate over each agent to find the best match
        for agent in self.agents:
            # Compute the embedding of the agent description
            agent_emb = self.get_embedding(agent["description"])

            if agent_emb is None:
                continue

            # Calculate cosine similarity between user prompt and agent description
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            print(similarity)

            # Select the agent with the highest similarity score
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        # Return error message if no suitable agent found
        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")

        # Return the response from the selected agent's function
        return best_agent["func"](user_input)
