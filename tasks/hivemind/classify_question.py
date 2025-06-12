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
    
    def classify_question_lm(self, message: str) -> QuestionClassificationResult:
        """
        Classify message using a language model to be a question or not
        Returns a QuestionClassificationResult with result and optionally reasoning
        """
        client = OpenAI()
        
        if self.enable_reasoning:
            user_prompt = (
                "Classify the following user message to be a question or not. "
                "First provide your reasoning, then give the result. "
                "Format your response as:\n"
                "Reasoning: [your detailed reasoning here]\n"
                "Result: [true or false]"
                f"\n\nMessage: {message}"
            )
        else:
            user_prompt = (
                "Classify the following user message to be a question or not. Reply with only a boolean value."
                f"\n\nMessage: {message}"
            )
        
        # Prepare chat completion parameters
        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a classification assistant that is very good at classifying messages to be a question or not."},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }
        
        response = client.chat.completions.create(**completion_params)
        response_text = response.choices[0].message.content.strip()
        
        # Parse response based on whether reasoning is enabled
        reasoning = None
        if self.enable_reasoning and "Reasoning:" in response_text and "Result:" in response_text:
            # Parse structured response
            try:
                parts = response_text.split("Result:")
                reasoning_part = parts[0].replace("Reasoning:", "").strip()
                result_part = parts[1].strip().lower()
                
                # Validate the result part
                if result_part in ["true", "yes", "1"]:
                    result = True
                elif result_part in ["false", "no", "0"]:
                    result = False
                else:
                    raise ValueError(f"Unexpected boolean response: '{result_part}'")
                
                reasoning = reasoning_part
            except (IndexError, ValueError):
                # Fallback to simple parsing if structured parsing fails
                response_lower = response_text.lower()
                if "true" in response_lower or "yes" in response_lower:
                    result = True
                elif "false" in response_lower or "no" in response_lower:
                    result = False
                else:
                    raise ValueError(f"Could not parse response: '{response_text}'")
        else:
            # Handle simple boolean response
            response_lower = response_text.lower()
            if response_lower in ["true", "yes", "1"]:
                result = True
            elif response_lower in ["false", "no", "0"]:
                result = False
            else:
                raise ValueError(f"Unexpected boolean response from model: '{response_text}'. Expected 'true' or 'false'.")
        
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

        if self.enable_reasoning:
            user_prompt = (
                """Assign a sensitivity score (0-1) to the following message according to the system rules. """
                """First provide your reasoning for the score, then give the numerical score. """
                """Format your response as:\n"""
                """Reasoning: [your detailed reasoning here]\n"""
                """Score: [numerical score between 0 and 1]"""
                f"""\n\nMessage: "{message}"""
            )
        else:
            user_prompt = (
                """Assign a sensitivity score (0-1) to the following message according to the system rules. Reply with only the number."""
                f"""\n\nMessage: "{message}"""
            )

        # Prepare chat completion parameters
        completion_params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
        }

        response = client.chat.completions.create(**completion_params)
        response_text = response.choices[0].message.content.strip()

        # Parse response based on whether reasoning is enabled
        reasoning = None
        if self.enable_reasoning and "Reasoning:" in response_text and "Score:" in response_text:
            # Parse structured response
            try:
                parts = response_text.split("Score:")
                reasoning_part = parts[0].replace("Reasoning:", "").strip()
                score_part = parts[1].strip()
                
                # Match any decimal number format
                if re.match(r"^-?\d*\.?\d+$", score_part):
                    score = float(score_part)
                    
                    # Validate score is between 0 and 1
                    if not (0 <= score <= 1):
                        raise ValueError(f"Generated score must be between 0 and 1, got: {score}")
                    
                    result = score >= self.rag_threshold
                else:
                    raise ValueError(f"Invalid score format: {score_part}")
                
                reasoning = reasoning_part
            except (IndexError, ValueError) as e:
                # Fallback to simple parsing if structured parsing fails
                response_lower = response_text.lower()
                if re.match(r"^-?\d*\.?\d+$", response_lower):
                    score = float(response_lower)
                    if not (0 <= score <= 1):
                        raise ValueError(f"Generated score must be between 0 and 1, got: {score}")
                    result = score >= self.rag_threshold
                else:
                    raise ValueError(f"Could not parse response: '{response_text}'")
        else:
            # Handle simple score response
            response_lower = response_text.lower()
            
            # Match any decimal number format (including negative and > 1)
            if re.match(r"^-?\d*\.?\d+$", response_lower):
                score = float(response_lower)
                
                # Validate score is between 0 and 1
                if not (0 <= score <= 1):
                    raise ValueError(f"Generated score must be between 0 and 1, got: {score}")
                
                result = score >= self.rag_threshold
            else:
                raise ValueError(f"Wrong response: {response_text}")
        
        # Prepare return data
        result_data = {"result": result, "score": score}
        if reasoning:
            result_data["reasoning"] = reasoning
        
        return MessageClassificationResult(**result_data)
