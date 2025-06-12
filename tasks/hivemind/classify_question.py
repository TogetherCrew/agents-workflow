import os
import re
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv
from transformers import pipeline
from pydantic import BaseModel


class QuestionClassificationResult(BaseModel):
    """Result of question classification"""
    result: bool
    reasoning: Optional[str] = None


class MessageClassificationResult(BaseModel):
    """Result of message classification for RAG"""
    result: bool
    score: float
    reasoning: Optional[str] = None


class ClassifyQuestion:
    def __init__(
            self,
            model: str = "gpt-4o-mini-2024-07-18",
            rag_threshold: float = 0.5,
            enable_reasoning: bool = False,
    ):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.enable_reasoning = enable_reasoning
        
        # Validate rag_threshold is between 0 and 1
        if not (0 <= rag_threshold <= 1):
            raise ValueError(f"rag_threshold must be between 0 and 1, got: {rag_threshold}")
        
        self.classification_model = "shahrukhx01/question-vs-statement-classifier"
        self.system_prompt = ("""You are a classification assistant. For any incoming user message, assign a sensitivity score between 0 and 1 reflecting how much it requires fetching up-to-date or specialized information from an external retrieval-augmented generation (RAG) data source."""
                                      """\n\nScoring guidelines:\n"""
                                      """- 1.0: definitely requires RAG (specific, dynamic, or domain-specific queries).\n"""
                                      """- 0.0: definitely does not (greetings, opinions, casual chat, or requests directed at a person).\n"""
                                      """- Intermediate values (e.g., 0.3, 0.7) indicate partial or borderline cases.\n"""
                                      """Apply these rules:\n"""
                                      """1. If the message asks for up-to-date or time-sensitive facts (e.g., "latest," "current price," "when X"), lean toward 1.0.\n"""
                                      """2. If it seeks definitions or explanations of specialized or domain-specific concepts, lean toward 1.0.\n"""
                                      """3. If it requests step-by-step procedures or how-to guides, lean toward 1.0.\n"""
                                      """4. If it needs project-, platform-, or asset-specific information (e.g., campaign status, token listings), lean toward 1.0.\n"""
                                      """5. If it asks for recommendations, comparisons, or "best" choices, lean toward 1.0.\n"""
                                      """6. If it involves legitimacy, validation, or security verification, lean toward 1.0.\n"""
                                      """7. If it references external identifiers, tickers, tokens, URLs, or passwords, lean toward 0.0.\n"""
                                      """8. If the answer cannot be derived from prior conversation context alone, lean toward 1.0.\n"""
                                      """9. If it's a greeting, opinion, speculation, brainstorming, or casual chat addressed to a person, lean toward 0.0.\n"""
                                      """10. For ambiguous or borderline cases, choose an appropriate fractional score between 0 and 1.\n"""
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
    
    def classify_question_lm(self, message: str) -> QuestionClassificationResult:
        """
        Classify message using a language model to be a question or not
        Returns a QuestionClassificationResult with result and optionally reasoning
        """
        client = OpenAI()
        
        user_prompt = (
            f"Classify the following user message to determine if it is a question or not.\n\nMessage: {message}"
        )
        
        # Define the response schema based on reasoning setting
        if self.enable_reasoning:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "question_classification_with_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "boolean",
                                "description": "Whether the message is a question (true) or not (false)"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for the classification decision"
                            }
                        },
                        "required": ["result", "reasoning"],
                        "additionalProperties": False
                    }
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "question_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "result": {
                                "type": "boolean",
                                "description": "Whether the message is a question (true) or not (false)"
                            }
                        },
                        "required": ["result"],
                        "additionalProperties": False
                    }
                }
            }
        
        # Prepare chat completion parameters
        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a classification assistant that is very good at classifying messages to be a question or not."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": response_format
        }
        
        response = client.chat.completions.create(**completion_params)
        response_text = response.choices[0].message.content.strip()
        
        # Parse the structured JSON response
        import json
        response_data = json.loads(response_text)
        
        result = bool(response_data["result"])
        reasoning = response_data.get("reasoning") if self.enable_reasoning else None
        
        # Prepare return data
        result_data = {"result": result}
        if reasoning:
            result_data["reasoning"] = reasoning
        
        return QuestionClassificationResult(**result_data)

    def classify_message_lm(self, message: str) -> MessageClassificationResult:
        """
        Classify message using a language model to be a RAG question or not
        Returns a MessageClassificationResult with result, score, and optionally reasoning
        """
        client = OpenAI()
        
        user_prompt = (
            f"""Assign a sensitivity score (0-1) to the following message according to the system rules.\n\nMessage: "{message}"""
        )

        # Define the response schema based on reasoning setting
        if self.enable_reasoning:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_classification_with_reasoning",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Sensitivity score between 0 and 1"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Explanation for the assigned score"
                            }
                        },
                        "required": ["score", "reasoning"],
                        "additionalProperties": False
                    }
                }
            }
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rag_classification",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                                "description": "Sensitivity score between 0 and 1"
                            }
                        },
                        "required": ["score"],
                        "additionalProperties": False
                    }
                }
            }

        # Prepare chat completion parameters
        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": response_format
        }

        response = client.chat.completions.create(**completion_params)
        response_text = response.choices[0].message.content.strip()

        # Parse the structured JSON response
        import json
        response_data = json.loads(response_text)
        
        score = float(response_data["score"])
        
        # Validate score is between 0 and 1 (should be enforced by schema, but double-check)
        if not (0 <= score <= 1):
            raise ValueError(f"Generated score must be between 0 and 1, got: {score}")
        
        result = score >= self.rag_threshold
        reasoning = response_data.get("reasoning") if self.enable_reasoning else None
        
        # Prepare return data
        result_data = {"result": result, "score": score}
        if reasoning:
            result_data["reasoning"] = reasoning
        
        return MessageClassificationResult(**result_data)
