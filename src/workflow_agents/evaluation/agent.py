from openai import OpenAI


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
