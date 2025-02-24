from dotenv import load_dotenv
import openai
import os


class CheckQuestion:
    def __init__(self, model: str):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.question_check_prompt = (
            """Determine if the user's message requires external knowledge from a RAG pipeline. 
            If yes, return **True**; otherwise, return **False**. However, if the user's request 
            is specifically directed to a person (even if no name is mentioned, e.g., “Could you 
            do this for me?”), always return **False**. Provide only “True” or “False,” with no 
            further explanation.
            
            Message: """
        )

    def check_question(self, message: str) -> bool:
        prompt = self.question_check_prompt + message
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            api_key=self.api_key
        )

        response_text = response['choices'][0]['message']['content'].strip().lower()

        if "true" in response_text:
            return True
        elif "false" in response_text:
            return False
        else:
            raise ValueError(f"Wrong response: {response_text}")
