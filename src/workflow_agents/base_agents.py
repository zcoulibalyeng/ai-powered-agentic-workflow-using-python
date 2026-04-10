from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from dotenv import load_dotenv


load_dotenv()


# DirectPromptAgent class definition
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

        

# AugmentedPromptAgent class definition
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



# KnowledgeAugmentedPromptAgent class definition
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


# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    A RAG (Retrieval-Augmented Generation) Knowledge Prompt Agent that uses
    embeddings to find relevant knowledge from a large corpus and responds
    based solely on retrieved information.

    This agent chunks text, calculates embeddings, and finds the most relevant
    chunk to answer user queries.

    Attributes:
        persona (str): The persona description for the agent.
        chunk_size (int): The size of text chunks for embedding.
        chunk_overlap (int): Overlap between consecutive chunks.
        openai_api_key (str): The API key for OpenAI authentication.
        unique_filename (str): Unique filename for storing chunks and embeddings.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initialize the RAGKnowledgePromptAgent.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            persona (str): The persona the agent should assume.
            chunk_size (int): The size of text chunks. Defaults to 2000.
            chunk_overlap (int): Overlap between chunks. Defaults to 100.
        """

        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def get_embedding(self, text):
        """
        Fetch the embedding vector for given text using OpenAI's embedding API.

        Args:
            text (str): Text to embed.

        Returns:
            list: The embedding vector.
        """
        client = OpenAI(api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculate cosine similarity between two vectors.

        Args:
            vector_one (list): First embedding vector.
            vector_two (list): Second embedding vector.

        Returns:
            float: Cosine similarity between vectors.
        """

        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        """
        Split text into manageable chunks, attempting natural breaks.

        Args:
            text (str): Text to split into chunks.

        Returns:
            list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        # If text is short, return it as one chunk
        if len(text) <= self.chunk_size:
            # return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
            chunks = [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]
            self._save_chunks_to_csv(chunks)
            return chunks

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end == len(text):
                break

            start = end - self.chunk_overlap
            chunk_id += 1
        self._save_chunks_to_csv(chunks)
        return chunks

        # with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
        #     writer.writeheader()
        #     for chunk in chunks:
        #         writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})
        #
        # return chunks

    def _save_chunks_to_csv(self, chunks):
        """Helper method to save chunks to CSV"""
        with open(f"chunks-{self.unique_filename}.csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size", "chunk_id", "start_char", "end_char"])
            writer.writeheader()
            for chunk in chunks:
                # Only write the keys that match our fieldnames
                filtered_chunk = {k: chunk.get(k) for k in ["text", "chunk_size", "chunk_id", "start_char", "end_char"]}
                writer.writerow(filtered_chunk)

    def calculate_embeddings(self):
        """
        Calculate embeddings for each chunk and store them in a CSV file.

        Returns:
            DataFrame: DataFrame containing text chunks and their embeddings.
        """

        df = pd.read_csv(f"chunks-{self.unique_filename}.csv", encoding='utf-8')
        df['embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}.csv", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Find and respond to a prompt based on similarity with embedded knowledge.

        Args:
            prompt (str): User input prompt.

        Returns:
            str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}.csv", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content


class EvaluationAgent:
    """
    An Evaluation Agent that assesses responses from a worker agent against
    specific criteria and refines responses through iterative feedback.

    This agent implements a loop that evaluates, provides corrections,
    and re-evaluates until criteria are met or max iterations reached.

    Attributes:
        openai_api_key (str): The API key for OpenAI authentication.
        persona (str): The persona for the evaluation agent.
        evaluation_criteria (str): The criteria to evaluate responses against.
        worker_agent: The agent whose responses will be evaluated.
        max_interactions (int): Maximum number of evaluation iterations.
        """
    
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        """
        Initialize the EvaluationAgent.

        Args:
            openai_api_key (str): The OpenAI API key for authentication.
            persona (str): The persona for the evaluation agent.
            evaluation_criteria (str): The criteria for evaluation.
            worker_agent: The agent to evaluate.
            max_interactions (int): Maximum evaluation iterations.
        """

        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        """
        Manage interactions between agents to achieve a solution meeting criteria.

        Args:
            initial_prompt (str): The initial prompt to evaluate.

        Returns:
            dict: Contains 'final_response', 'evaluation', and 'iterations'.
        """

        client = OpenAI(api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = ""

        for i in range(self.max_interactions):
            print(f"\n--- Interaction {i+1} ---")

            # Step 1: Worker agent generates a response to the prompt
            print(" Step 1: Worker agent generates a response to the prompt")

            print(f"Prompt:\n{prompt_to_evaluate}")

            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)

            print(f"Worker Agent Response:\n{response_from_worker}")

            # Step 2: Evaluator agent judges the response
            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"  
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )

            # Define message structure for evaluation (temperature=0)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"{self.persona}"},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")


            # Step 3: Check if evaluation is positive
            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("✅ Final solution accepted.")
                break
            else:
                # Step 4: Generate instructions to correct the response
                print(" Step 4: Generate instructions to correct the response")
                instruction_prompt = (
                    f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                )

                # Define message structure for correction instructions (temperature=0)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages= [
                        {"role": "system", "content": "You are a helpful assistant that provides clear correction instructions."},
                        {"role": "user", "content": instruction_prompt}
                    ],
                    temperature=0
                )
                instructions = response.choices[0].message.content.strip()
                print(f"Instructions to fix:\n{instructions}")


                # Step 5: Send feedback to worker agent for refinement
                print(" Step 5: Send feedback to worker agent for refinement")
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )

        # Return dictionary with results
        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": i + 1
        }   


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
