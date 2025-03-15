import os

from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline


class ClassifyQuestion:
    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.classification_model = "shahrukhx01/question-vs-statement-classifier"
        self.question_check_prompt = """Determine if the user's message requires external knowledge from a RAG pipeline. 
            If yes, return **True**; otherwise, return **False**. However, if the user's request 
            is specifically directed to a person (even if no name is mentioned, e.g., “Could you 
            do this for me?”), always return **False**. Provide only “True” or “False,” with no 
            further explanation.
            
            Message: """

    def classify_message(self, message: str) -> bool:
        """
        classify if a message is a question or statement using a local model
        """
        custom_labels = {
            "LABEL_0": False,  # STATEMENT
            "LABEL_1": True,  # QUESTION
        }

        pipe = pipeline("text-classification", model=self.classification_model)
        out = pipe(message)
        is_question = custom_labels.get(out[0]["label"])
        return is_question

    def classify_message_lm(self, message: str) -> bool:
        """
        Classify message using a language model
        """
        client = OpenAI()
        prompt = self.question_check_prompt + message

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )

        response_text = response.choices[0].message.content.strip().lower()

        if "true" in response_text:
            return True
        elif "false" in response_text:
            return False
        else:
            raise ValueError(f"Wrong response: {response_text}")
