import os
from dotenv import load_dotenv
from openai import OpenAI

from pydantic import BaseModel


class AnswerValidator:
    def __init__(self, model: str = "gpt-4o-mini-2024-07-18"):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")

        class ValidatorSchema(BaseModel):
            relative: bool

        self.validator_model = ValidatorSchema

    def check_answer_validity(self, question: str, answer: str) -> bool:
        client = OpenAI()
        response = client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that validates the if the answer is relative to the question or not.",
                },
                {
                    "role": "user",
                    "content": f"**Question:** {question}\n\n**Answer:** {answer}",
                },
            ],
            response_format=self.validator_model,
            temperature=0.0,
        )
        return response.choices[0].message.parsed.relative
