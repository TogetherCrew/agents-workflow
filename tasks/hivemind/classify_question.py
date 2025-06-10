import os
import re

from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline


class ClassifyQuestion:
    def __init__(
            self,
            model: str = "gpt-4o-mini-2024-07-18",
            rag_threshold: float = 0.5,
    ):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.classification_model = "shahrukhx01/question-vs-statement-classifier"
        self.system_prompt = ("""You are a classification assistant. For any incoming user message, assign a sensitivity score between 0 and 1 reflecting how much it requires fetching up-to-date or specialized information from an external retrieval-augmented generation (RAG) data source and provide a reasoning for your score."""
                                      """\n\nScoring guidelines:\n"""
                                      """- 1.0: definitely requires RAG (specific, dynamic, or domain-specific queries).\n"""
                                      """- 0.0: definitely does not (greetings, opinions, casual chat, or requests directed at a person).\n"""
                                      """- Intermediate values (e.g., 0.3, 0.7) indicate partial or borderline cases.\n"""
                                      """Apply these rules:\n"""
                                      """1. If the message asks for up-to-date or time-sensitive facts (e.g., “latest,” “current price,” “when X”), lean toward 1.0.\n"""
                                      """2. If it seeks definitions or explanations of specialized or domain-specific concepts, lean toward 1.0.\n"""
                                      """3. If it requests step-by-step procedures or how-to guides, lean toward 1.0.\n"""
                                      """4. If it needs project-, platform-, or asset-specific information (e.g., campaign status, token listings), lean toward 1.0.\n"""
                                      """5. If it asks for recommendations, comparisons, or “best” choices, lean toward 1.0.\n"""
                                      """6. If it involves legitimacy, validation, or security verification, lean toward 1.0.\n"""
                                      """7. If it references external identifiers, tickers, tokens, URLs, or passwords, lean toward 0.0.\n"""
                                      """8. If the answer cannot be derived from prior conversation context alone, lean toward 1.0.\n"""
                                      """9. If it's a greeting, opinion, speculation, brainstorming, or casual chat addressed to a person, lean toward 0.0.\n"""
                                      """10. For ambiguous or borderline cases, choose an appropriate fractional score between 0 and 1.\n"""
                                      """\nRespond with exactly one decimal number between 0 and 1 (e.g., `0`, `0.5`, `1`). No extra text."""
        )
        self.rag_threshold = rag_threshold

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
        user_prompt = (
            """Assign a sensitivity score (0-1) to the following message according to the system rules. Reply with only the number."""
            f"""\n\nMessage: "{message}"""
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content.strip().lower()

        if re.match(r"^(?:0|1|\d+\.\d+)$", response_text):
            return float(response_text) >= self.rag_threshold
        else:
            raise ValueError(f"Wrong response: {response_text}")
